import os
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
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

def log_memory_usage(tag=""):
    """Log GPU memory usage at a specific point in the code"""
    memory_allocated = torch.cuda.memory_allocated() / 1024**2
    memory_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"MEMORY [{tag}] Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")

# Create a callback for memory tracking during training
class MemoryTrackingCallback(TrainerCallback):
    def __init__(self, log_freq=10):
        self.log_freq = log_freq
    
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq == 0:
            log_memory_usage(f"step{state.global_step}_begin")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_freq == 0:
            log_memory_usage(f"step{state.global_step}_end")
            # Clear cache periodically to reduce fragmentation
            if state.global_step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                log_memory_usage(f"step{state.global_step}_after_cache_clear")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        log_memory_usage(f"epoch{state.epoch}_begin")
        print_gpu_memory_stats()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        log_memory_usage(f"epoch{state.epoch}_end")
        print_gpu_memory_stats()
    
    def on_train_begin(self, args, state, control, **kwargs):
        log_memory_usage("train_begin")
        print_gpu_memory_stats()

# Custom dataset processing
def prepare_dataset(batch, processor):
    # Load audio and process it
    audio = batch["audio"]
    
    # Process audio data
    batch["input_features"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    
    # Process text to get labels
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch

def main():
    # Set visible GPUs if needed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Track memory at beginning of script
    log_memory_usage("init")
    print_gpu_memory_stats(local_rank)
    
    # Load Whisper model
    model_name = "openai/whisper-medium"
    print(f"Loading model: {model_name}")
    
    # Load processor first to separate memory usage
    processor = WhisperProcessor.from_pretrained(model_name)
    log_memory_usage("after_processor_load")
    
    # Load model with efficient memory options
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,  # Use fp16 to reduce memory usage
        low_cpu_mem_usage=True,
    )
    log_memory_usage("after_model_load")
    
    # Enable gradient checkpointing for better memory efficiency
    model.gradient_checkpointing_enable()
    
    # Clear cache to free up memory
    gc.collect()
    torch.cuda.empty_cache()
    log_memory_usage("after_cache_clear")
    
    # Load a dataset - using Common Voice as an example
    try:
        # Attempt to load Common Voice dataset
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="train[:50]", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load Common Voice dataset: {e}")
        print("Falling back to LibriSpeech dataset")
        dataset = load_dataset("librispeech_asr", "clean", split="train.100[:50]", trust_remote_code=True)
    
    # Ensure audio format is consistent
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Process the dataset
    processed_dataset = dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=dataset.column_names,
        num_proc=2,  # Adjust based on CPU cores
    )
    
    # Define the data collator
    from dataclasses import dataclass
    from typing import Dict, List, Union
    
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: WhisperProcessor
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Process input features
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            # Process labels
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            # Replace padding with -100 for loss computation
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            # Add labels to batch
            batch["labels"] = labels
            
            return batch
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="whisper-medium-finetuned",
        per_device_train_batch_size=8,  # Start small, increase if memory allows
        gradient_accumulation_steps=2,  # Increase effective batch size
        learning_rate=5e-5,
        warmup_steps=10,
        max_steps=30,
        fp16=False,  # Use mixed precision
        logging_steps=10,
        save_steps=10,
        eval_steps=10,
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        remove_unused_columns=False,  # Important for custom features
        label_names=["labels"],
        report_to="none",  # Disable wandb/tensorboard for memory efficiency
        # Distributed training settings
        local_rank=local_rank,
        ddp_find_unused_parameters=False,  # Set to True if needed
        # OOM prevention
        gradient_checkpointing=True,
        optim="adamw_torch",  # Can also try "adamw_8bit" with bitsandbytes installed
    )
    
    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset.select(range(min(50, len(processed_dataset)))),  # Small eval dataset
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[MemoryTrackingCallback()],
    )
    
    # Try to detect potential OOM before training
    try:
        # Monitor memory with a dry run
        print("\nPerforming memory usage estimation...")
        # Create a small subset to test memory usage
        mini_dataset = processed_dataset.select(range(min(10, len(processed_dataset))))
        test_dataloader = trainer.get_train_dataloader()
        for idx, batch in enumerate(test_dataloader):
            if idx >= 2:  # Just check first two batches
                break
            print(f"Test batch {idx+1}/2")
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: {val.shape}, {val.dtype}")
            # Simulate a forward pass to check memory
            with torch.no_grad():
                outputs = model(**{k: v.to(trainer.args.device) for k, v in batch.items() if k != "labels"})
            log_memory_usage(f"test_batch{idx+1}_forward")
            del outputs
            torch.cuda.empty_cache()
        print("Memory estimation complete!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM detected during test run!")
            print("Recommendations:")
            print("1. Reduce batch size (currently", training_args.per_device_train_batch_size, ")")
            print("2. Increase gradient accumulation (currently", training_args.gradient_accumulation_steps, ")")
            print("3. Try loading in 8-bit with bitsandbytes")
            print("4. Consider further input sequence length limitations")
            print_gpu_memory_stats()
            raise e
    
    # Start training with OOM handling
    try:
        print("\nStarting training...")
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM ERROR during training!")
            print_gpu_memory_stats()
            
            print("\nOOM Troubleshooting:")
            print("1. Current settings:")
            print(f"   - Batch size: {training_args.per_device_train_batch_size}")
            print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
            print(f"   - Mixed precision: {training_args.fp16}")
            print(f"   - Gradient checkpointing: {training_args.gradient_checkpointing}")
            
            print("\n2. Try these adjustments:")
            print("   - Reduce batch size to 1")
            print("   - Increase gradient accumulation to 16")
            print("   - Install bitsandbytes and use 8-bit optimizers")
            print("   - Pre-process audio to limit max length")
            print("   - Use DeepSpeed ZeRO optimization")
            
            # Attempt to identify the specific operation causing OOM
            error_context = str(e).split('\n')
            for line in error_context:
                if 'size' in line or 'shape' in line:
                    print(f"\nPotential issue: {line}")
            
            raise e

if __name__ == "__main__":
    main()