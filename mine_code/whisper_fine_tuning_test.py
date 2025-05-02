import os
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Audio
import gc
import GPUtil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# Memory tracking functions
def print_gpu_memory_stats(device=None):
    """Print detailed GPU memory statistics for specified device or all devices."""
    nvmlInit()
    if device is None:
        devices = range(torch.cuda.device_count())
    else:
        devices = [device]
    
    for i in devices:
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i} Memory: Used {info.used/1024**2:.2f}MB / Total {info.total/1024**2:.2f}MB")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i} PyTorch: Allocated {allocated:.2f}MB / Reserved {reserved:.2f}MB")

def log_memory_usage(tag="", rank=0):
    """Log GPU memory usage at a specific point in the code"""
    memory_allocated = torch.cuda.memory_allocated(rank) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(rank) / 1024**2
    print(f"GPU {rank} MEMORY [{tag}] Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")

def init_distributed():
    """Initialize distributed training setup"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Distributed environment variables not set. Using defaults.")
        rank = 0
        world_size = 1
        local_rank = 0
        
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return local_rank, rank, world_size

class AudioDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load audio and preprocess in a memory-efficient way
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        
        # Process audio data
        input_features = self.processor(
            audio, 
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze()
        
        # Get the corresponding labels
        # Different datasets use different column names for transcriptions
        if "text" in item:
            text = item["text"]
        elif "sentence" in item:
            text = item["sentence"]
        elif "transcription" in item:
            text = item["transcription"]
        else:
            # Fallback for Common Voice which uses "sentence" in newer versions
            for possible_text_column in ["sentence", "transcript", "normalized_text"]:
                if possible_text_column in self.dataset.column_names:
                    text = item.get(possible_text_column, "")
                    break
            else:
                # If no text column is found, use empty string
                print(f"Warning: No text column found in dataset. Available columns: {list(item.keys())}")
                text = ""
        
        labels = self.processor.tokenizer(text).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }

def collate_fn(batch):
    """Custom collate function to handle variable-length inputs"""
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad input features
    max_len = max([item.shape[0] for item in input_features])
    padded_inputs = []
    
    for item in input_features:
        if item.shape[0] < max_len:
            # Pad with zeros
            padding = torch.zeros((max_len - item.shape[0], item.shape[1]))
            padded_item = torch.cat([item, padding], dim=0)
            padded_inputs.append(padded_item)
        else:
            padded_inputs.append(item)
    
    # Convert to tensors
    input_features = torch.stack(padded_inputs)
    
    # Pad labels
    max_label_len = max([len(label) for label in labels])
    padded_labels = []
    
    for label in labels:
        if len(label) < max_label_len:
            padded_label = label + [-100] * (max_label_len - len(label))
            padded_labels.append(padded_label)
        else:
            padded_labels.append(label)
    
    labels = torch.tensor(padded_labels)
    
    return {
        "input_features": input_features,
        "labels": labels
    }

def main():
    # Initialize distributed training
    local_rank, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"Starting process rank {rank} on device {device}")
    
    # Track memory at beginning of script
    log_memory_usage("init", local_rank)
    print_gpu_memory_stats(local_rank)
    
    # Load Whisper model
    model_name = "openai/whisper-large-v3"
    print(f"Loading model: {model_name}")
    
    # Load processor first to separate memory usage
    processor = WhisperProcessor.from_pretrained(model_name)
    log_memory_usage("after_processor_load", local_rank)
    
    # Load model with efficient memory options
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16  # Use fp16 to reduce memory usage
    )
    log_memory_usage("after_model_load", local_rank)
    print_gpu_memory_stats(local_rank)
    
    # Clear cache to free up memory
    gc.collect()
    torch.cuda.empty_cache()
    log_memory_usage("after_cache_clear", local_rank)
    
    # Load a small audio dataset
    # Using Common Voice as an example
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="train[:50]", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split dataset for distributed training
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
    
    # Create DataLoader with distributed sampler
    train_dataset = AudioDataset(dataset, processor)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Memory-efficient batch size - start small and increase if possible
    batch_size = 2  # Start with a small batch size
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,  # Adjust based on your CPU
        pin_memory=True
    )
    
    log_memory_usage("after_dataloader_setup", local_rank)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank])
    log_memory_usage("after_ddp_setup", local_rank)
    
    # Setup optimizer with gradient accumulation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    
    # Gradient accumulation steps (adjust based on memory availability)
    gradient_accumulation_steps = 2
    
    # Learning rate scheduler
    total_steps = len(dataloader) // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop with memory tracking
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            # Check memory before forward pass
            if step % 10 == 0 and rank == 0:
                log_memory_usage(f"epoch{epoch}_batch{step}_before_forward", local_rank)
            
            # Move batch to device
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            try:
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Memory tracking after step
                    if rank == 0:
                        log_memory_usage(f"epoch{epoch}_step{step}_after_optim", local_rank)
                
                # Print progress
                running_loss += loss.item() * gradient_accumulation_steps
                if step % 10 == 0 and rank == 0:
                    print(f"Step {step}/{len(dataloader)}, Loss: {running_loss/(step+1):.4f}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM ERROR on rank {rank} at step {step}!")
                    print_gpu_memory_stats(local_rank)
                    
                    # Provide troubleshooting guidance
                    print("Troubleshooting suggestions:")
                    print("1. Reduce batch size (current:", batch_size, ")")
                    print("2. Increase gradient accumulation steps (current:", gradient_accumulation_steps, ")")
                    print("3. Try sequence bucketing to reduce padding")
                    print("4. Further reduce precision (e.g., use bfloat16 if available)")
                    
                    # Free memory
                    del input_features, labels, outputs, loss
                    torch.cuda.empty_cache()
                    raise e
        
        # Save checkpoint at end of epoch (only from rank 0)
        if rank == 0 and epoch % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f"whisper_medium_checkpoint_epoch{epoch}.pt")
            print(f"Checkpoint saved for epoch {epoch+1}")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()