### ========================================================= ###
### whipser fine tuning custom training loop, without Trainer
### ========================================================= ###
import torch
import torchaudio
import pynvml
from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import numpy as np
from torch.utils.data import DataLoader
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast, GradScaler
import gc
from tqdm import tqdm

# GPU memory tracking
def get_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / 1024**2
    total = info.total / 1024**2
    pynvml.nvmlShutdown()
    return used, total

# Clear GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f"Initial GPU Memory: {get_gpu_memory()} MB (Used/Total)")
print(f"PyTorch Version: {torch.__version__}, CUDA Version: {torch.version.cuda}")

dataset = DatasetDict({
    "train": load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="train[:100]", trust_remote_code=True),
    "validation": load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="validation[:100]", trust_remote_code=True),
    "test": load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="test[:100]", trust_remote_code=True)
})

model = "openai/whisper-medium"
# Load processor
processor = WhisperProcessor.from_pretrained(model, language="ja", task="transcribe")

# Preprocess function
def preprocess(batch):
    audio = batch["audio"]
    # Resample to 16kHz
    waveform = torchaudio.load(audio["path"])[0]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    resampler = torchaudio.transforms.Resample(audio["sampling_rate"], 16000)
    audio_array = resampler(waveform).squeeze().numpy()
    
    # Filter audio length (2s to 30s)
    if 2 <= len(audio_array) / 16000 <= 30:
        batch["input_features"] = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features[0]
        batch["labels"] = processor.tokenizer.encode(batch["sentence"], return_tensors="pt")[0]
    else:
        batch["input_features"] = None
        batch["labels"] = None
    return batch

# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=["audio", "sentence", "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

# Filter out None values
def filter_none(batch):
    return batch["input_features"] is not None and batch["labels"] is not None

dataset = dataset.filter(filter_none)

print(f"Dataset sizes: Train={len(dataset['train'])}, Validation={len(dataset['validation'])}, Test={len(dataset['test'])}")
print(f"GPU Memory after preprocessing: {get_gpu_memory()} MB")

def collate_fn(features):
    input_features = [f["input_features"] for f in features]
    labels = [f["labels"] for f in features]
    
    input_features = processor.feature_extractor.pad({"input_features": input_features}, return_tensors="pt")["input_features"]
    labels = processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
    labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
    
    return {"input_features": input_features, "labels": labels}

clear_gpu_memory()
model = WhisperForConditionalGeneration.from_pretrained(model)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")
model = model.to("cuda")
model.gradient_checkpointing_enable()
print(f"GPU Memory after model loading: {get_gpu_memory()} MB")

wer_metric = evaluate.load("wer")

def compute_wer(predictions, labels):
    predictions[predictions == -100] = processor.tokenizer.pad_token_id
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return wer_metric.compute(predictions=pred_str, references=label_str)

# Hyperparameters
batch_size = 1
gradient_accumulation_steps = 2
learning_rate = 1e-5
max_steps = 10
eval_steps = 5
warmup_steps = 5
output_dir = "./whisper-large-ja-finetuned"

# Optimizer and scaler
optimizer = AdamW8bit(model.parameters(), lr=learning_rate)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

# DataLoaders
train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Training loop
model.train()
step = 0
best_wer = float("inf")
clear_gpu_memory()
print(f"GPU Memory before training: {get_gpu_memory()} MB")

while step < max_steps:
    progress_bar = tqdm(train_loader, desc=f"Step {step}/{max_steps}")
    accum_loss = 0
    for batch_idx, batch in enumerate(progress_bar):
        if step >= max_steps:
            break
        
        input_features = batch["input_features"].to("cuda")
        labels = batch["labels"].to("cuda")
        
        with autocast():
            outputs = model(input_features, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        accum_loss += loss.item() * gradient_accumulation_steps
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            
            torch.cuda.synchronize()
            progress_bar.set_postfix({"loss": accum_loss / gradient_accumulation_steps})
            accum_loss = 0
            
            # Evaluation
            if step % eval_steps == 0:
                model.eval()
                val_wer = 0
                val_steps = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        input_features = val_batch["input_features"].to("cuda")
                        labels = val_batch["labels"].to("cuda")
                        outputs = model.generate(input_features)
                        val_wer += compute_wer(outputs, labels)
                        val_steps += 1
                val_wer /= val_steps
                print(f"Step {step}, Validation WER: {val_wer:.4f}")
                
                # Save best model
                if val_wer < best_wer:
                    best_wer = val_wer
                    model.save_pretrained(f"{output_dir}/best")
                    processor.save_pretrained(f"{output_dir}/best")
                
                model.train()
                clear_gpu_memory()
        
        # Memory check
        if (batch_idx + 1) % 10 == 0:
            print(f"GPU Memory at step {step}, batch {batch_idx}: {get_gpu_memory()} MB")
    
    if step >= max_steps:
        break

print(f"GPU Memory after training: {get_gpu_memory()} MB")

### ========================================================= ###
### whipser fine tuning with pytorch lightning deepspeed
### ========================================================= ###

import os
import sys
import time
import json
import logging
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple

import torch
import psutil
import pytorch_lightning as pl
import evaluate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('whisper_training.log')
    ]
)
logger = logging.getLogger('whisper_finetuning')

@dataclass
class MemoryStats:
    """Track per-GPU memory statistics."""
    allocated_per_gpu: Dict[int, float] = field(default_factory=dict)
    reserved_per_gpu: Dict[int, float] = field(default_factory=dict)
    peak_per_gpu: Dict[int, float] = field(default_factory=dict)
    utilization_alert: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert memory stats to dictionary for logging."""
        return {
            "allocated_per_gpu": self.allocated_per_gpu,
            "reserved_per_gpu": self.reserved_per_gpu,
            "peak_per_gpu": self.peak_per_gpu,
            "utilization_alert": self.utilization_alert,
            "timestamp": self.timestamp
        }

class MemoryTracker:
    """Efficient memory tracking for distributed training."""
    
    def __init__(
        self,
        log_to_file: bool = False,
        log_path: str = "memory_stats.json",
        alert_threshold: float = 0.85,
        log_frequency: int = 10,
    ):
        self.log_to_file = log_to_file
        self.log_path = log_path
        self.alert_threshold = alert_threshold
        self.log_frequency = log_frequency
        self.memory_log: List[Dict] = []
        self.peak_stats = MemoryStats()
        
        # Get available GPU count (limit to actual devices)
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Initialize peak stats for available GPUs
        if torch.cuda.is_available():
            for i in range(self.device_count):
                self.peak_stats.peak_per_gpu[i] = 0.0

    @rank_zero_only
    def log_memory_usage(self, tag: str = "") -> Optional[MemoryStats]:
        """Track and log per-GPU memory usage (only on rank 0)."""
        if not torch.cuda.is_available():
            return None
            
        stats = MemoryStats()
        
        try:
            for i in range(self.device_count):
                # Get memory stats for this GPU
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                max_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                
                # Update stats
                stats.allocated_per_gpu[i] = allocated
                stats.reserved_per_gpu[i] = reserved
                stats.peak_per_gpu[i] = max(
                    self.peak_stats.peak_per_gpu.get(i, 0.0),
                    allocated
                )
                
                # Check if memory usage exceeds threshold
                if allocated / max_memory > self.alert_threshold:
                    stats.utilization_alert = True
                    logger.warning(
                        f"[{tag}] GPU {i} High Memory Usage: "
                        f"{allocated:.2f}/{max_memory:.2f} GB "
                        f"({allocated/max_memory*100:.1f}%)"
                    )
                
                # Update peak stats
                self.peak_stats.peak_per_gpu[i] = stats.peak_per_gpu[i]
            
            # Log detailed stats if there's an alert
            if stats.utilization_alert:
                for i in range(self.device_count):
                    logger.info(
                        f"[{tag}] GPU {i}: Allocated {stats.allocated_per_gpu[i]:.2f} GB | "
                        f"Reserved {stats.reserved_per_gpu[i]:.2f} GB | "
                        f"Peak {stats.peak_per_gpu[i]:.2f} GB"
                    )
            
            # Write to log file if enabled
            if self.log_to_file:
                log_entry = {"tag": tag, "stats": stats.to_dict()}
                self.memory_log.append(log_entry)
                
                # Periodically save to disk to avoid frequent I/O
                if len(self.memory_log) % self.log_frequency == 0:
                    self._save_log()
            
            return stats
            
        except Exception as e:
            logger.error(f"[{tag}] Memory tracking failed: {str(e)}")
            return None
    
    def _save_log(self) -> None:
        """Save memory log to disk."""
        if not self.log_to_file or not self.memory_log:
            return
            
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.memory_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory log: {str(e)}")
    
    def empty_cache(self, tag: str = "") -> None:
        """Empty CUDA cache and run garbage collection efficiently."""
        if not torch.cuda.is_available():
            return
        
        # Run garbage collection first to free Python objects
        gc.collect()
        
        try:
            # Get local rank (or use 0 if not distributed)
            local_rank = 0
            if torch.distributed.is_initialized():
                local_rank = torch.distributed.get_rank()
            
            # Only track memory if this is a device we're using
            if local_rank < self.device_count:
                # Measure memory before clearing
                before_reserved = torch.cuda.memory_reserved(local_rank) / (1024 ** 3)
                
                # Clear cache
                torch.cuda.empty_cache()
                
                # Measure memory after clearing
                after_reserved = torch.cuda.memory_reserved(local_rank) / (1024 ** 3)
                
                # Calculate and log freed memory
                freed = max(0.0, before_reserved - after_reserved)
                if freed > 0 and (not torch.distributed.is_initialized() or local_rank == 0):
                    logger.info(f"[{tag}] GPU {local_rank} cleanup freed: {freed:.2f} GB")
                    
        except Exception as e:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.error(f"[{tag}] Cache clearing failed: {str(e)}")
    
    def __del__(self):
        """Save log on destruction if needed."""
        if self.log_to_file and self.memory_log:
            self._save_log()


# Initialize global memory tracker
memory_tracker = MemoryTracker(log_path="whisper_memory_stats.json")

class JapaneseWhisperDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processor: WhisperProcessor,
        batch_size: int = 8,
        num_workers: int = 4,
        max_duration: float = 30.0,
        sample_rate: int = 16000,
        dataset_name: str = "mozilla-foundation/common_voice_11_0",
        dataset_config: str = "ja",
        train_split: str = "train[:100]",
        val_split: str = "validation[:100]",
        test_split: str = "test[:100]"
    ):
        super().__init__()
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.splits = {"train": train_split, "val": val_split, "test": test_split}
        self.datasets = {}

    def setup(self, stage: Optional[str] = None):
        # Load all splits at once to avoid redundant downloads
        datasets = load_dataset(
            self.dataset_name,
            self.dataset_config,
            trust_remote_code=True,
            split=list(self.splits.values())
        )

        # Pre-define filtering and feature preparation functions
        def is_audio_valid(sample):
            try:
                return sample["audio"]["array"].shape[0] / self.sample_rate <= self.max_duration
            except Exception:
                return False

        def prepare_features(batch):
            try:
                audio = batch["audio"]
                features = self.processor.feature_extractor(
                    audio["array"], sampling_rate=self.sample_rate
                )
                return {
                    "input_features": features.input_features[0],
                    "labels": self.processor.tokenizer(batch["sentence"]).input_ids
                }
            except Exception:
                # Return zero tensor as placeholder for failed processing
                return {"input_features": torch.zeros(80, 3000), "labels": []}

        # Process each split
        for split_name, dataset in zip(self.splits.keys(), datasets):
            # Apply processing pipeline
            processed_dataset = (
                dataset.select_columns(["audio", "sentence"])
                .cast_column("audio", Audio(sampling_rate=self.sample_rate))
                .filter(is_audio_valid)
                .map(
                    prepare_features, 
                    remove_columns=["audio", "sentence"],
                    num_proc=min(self.num_workers, 4) if self.num_workers > 0 else None
                )
            )
            
            self.datasets[split_name] = processed_dataset
            logger.info(f"{split_name.capitalize()} dataset size: {len(processed_dataset)}")

    def collate_fn(self, batch):
        # Handle empty batches gracefully
        if not batch:
            return {"input_features": torch.zeros(0, 80, 3000), "labels": torch.zeros(0, 0).long()}
        
        # Separate features and labels
        input_features = [{"input_features": x["input_features"]} for x in batch]
        labels = [{"input_ids": x["labels"]} for x in batch]
        
        # Pad batches
        batch_features = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch_labels = self.processor.tokenizer.pad(labels, return_tensors="pt")
        
        # Replace padding token id with -100 for loss calculation
        batch_labels["input_ids"][batch_labels["input_ids"] == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "input_features": batch_features["input_features"],
            "labels": batch_labels["input_ids"]
        }

    def _get_dataloader(self, split_name, shuffle=False):
        """Centralized dataloader creation to reduce code duplication"""
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=(split_name == "train"),  # Only drop last for training
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self):
        return self._get_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")


class WhisperJapaneseFineTuner(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        processor: Optional[WhisperProcessor] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        freeze_encoder: bool = False,
        max_new_tokens: int = 225,
        language: str = "ja",
        task: str = "transcribe"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        
        # Load model and processor
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.processor = processor or WhisperProcessor.from_pretrained(model_id, language=language, task=task)
        
        # Set forced decoder IDs for Japanese transcription
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.model.get_encoder().parameters():
                param.requires_grad = False

        # Load metrics
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def forward(self, input_features, labels=None):
        return self.model(input_features=input_features, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_features"], batch["labels"])
        self.log("train/loss", outputs.loss, prog_bar=True, sync_dist=True, batch_size=len(batch["labels"]))
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        # Calculate loss
        outputs = self(batch["input_features"], batch["labels"])
        self.log("val/loss", outputs.loss, prog_bar=True, sync_dist=True, batch_size=len(batch["labels"]))

        # Generate predictions
        predicted_ids = self.model.generate(
            input_features=batch["input_features"],
            max_new_tokens=self.hparams.max_new_tokens
        )
        
        # Decode predictions and references
        transcriptions = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Clean labels (replace -100 with pad token ID)
        labels_cleaned = batch["labels"].clone()
        labels_cleaned[labels_cleaned == -100] = self.processor.tokenizer.pad_token_id
        references = self.processor.tokenizer.batch_decode(labels_cleaned, skip_special_tokens=True)

        # Calculate metrics
        wer = self.wer_metric.compute(predictions=transcriptions, references=references)
        cer = self.cer_metric.compute(predictions=transcriptions, references=references)
        
        # Log metrics
        self.log("val/wer", wer, prog_bar=True, sync_dist=True, batch_size=len(batch["labels"]))
        self.log("val/cer", cer, prog_bar=True, sync_dist=True, batch_size=len(batch["labels"]))

        # Log samples from the first batch on rank 0
        if batch_idx == 0 and self.global_rank == 0:
            for i in range(min(2, len(transcriptions))):
                logger.info(f"Val example {i}: Pred: '{transcriptions[i]}' | True: '{references[i]}'")
                
        return {"wer": wer, "cer": cer}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        # Optional: Add any test epoch end logic here
        pass

    def configure_optimizers(self):
        # Define parameter groups with different weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate,
            eps=1e-8
        )
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * self.hparams.warmup_ratio),
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def create_deepspeed_config(
    stage: int = 2, 
    offload: bool = False, 
    gradient_accumulation: int = 2,
    fp16: bool = True,
    bf16: bool = False
) -> Dict:
    """Create DeepSpeed configuration with appropriate settings."""
    config = {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": gradient_accumulation,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto"
        }
    }
    
    # Configure precision
    if fp16:
        config["fp16"] = {"enabled": True, "auto_cast": True}
    elif bf16:
        config["bf16"] = {"enabled": True}
    
    # Configure offloading
    if offload and stage >= 2:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu", 
            "pin_memory": True, 
            "buffer_count": 4,
            "fast_init": True
        }
        if stage == 3:
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu", 
                "pin_memory": True
            }
    
    return config


def train_whisper_model(
    model_id: str = "openai/whisper-small",
    output_dir: str = "whisper-ja-finetuned",
    batch_size: int = 2,
    num_workers: int = 0,
    learning_rate: float = 3e-5,
    max_epochs: int = 3,
    deepspeed_stage: int = 2,
    offload_optimizer: bool = False,
    gradient_accumulation: int = 2,
    precision: str = "16-mixed",
    seed: int = 42,
    train_split: str = "train[:100]",
    val_split: str = "validation[:100]",
    test_split: str = "test[:100]",
    freeze_encoder: bool = False,
    language: str = "ja",
    task: str = "transcribe",
    max_new_tokens: int = 225,
    resume_from_checkpoint: Optional[str] = None
):
    """
    Train a Whisper model for Japanese ASR with optimized distributed training.
    
    Args:
        model_id: Base Whisper model ID to fine-tune
        output_dir: Directory to save model, logs, and checkpoints
        batch_size: Batch size per GPU
        num_workers: Number of dataloader workers
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum training epochs
        deepspeed_stage: DeepSpeed ZeRO stage (0, 1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
        gradient_accumulation: Gradient accumulation steps
        precision: Training precision (16-mixed, bf16-mixed, 32-true)
        seed: Random seed for reproducibility
        train_split: Training dataset split specification
        val_split: Validation dataset split specification
        test_split: Test dataset split specification
        freeze_encoder: Whether to freeze the encoder parameters
        language: Target language for transcription
        task: Task type (transcribe, translate)
        max_new_tokens: Maximum new tokens for generation
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Set random seed for reproducibility
    pl.seed_everything(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(output_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info(f"Starting training with model: {model_id}")

    # Initialize processor
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    
    # Create DeepSpeed configuration
    use_bf16 = "bf16" in precision
    ds_config = create_deepspeed_config(
        deepspeed_stage, 
        offload_optimizer, 
        gradient_accumulation, 
        fp16=not use_bf16,
        bf16=use_bf16
    )
    
    # Save DeepSpeed config
    ds_config_path = os.path.join(output_dir, "ds_config.json")
    with open(ds_config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    logger.info(f"DeepSpeed config saved to {ds_config_path}")

    # Create data module
    data_module = JapaneseWhisperDataModule(
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split
    )

    # Create model
    model = WhisperJapaneseFineTuner(
        model_id=model_id,
        processor=processor,
        learning_rate=learning_rate,
        freeze_encoder=freeze_encoder,
        max_new_tokens=max_new_tokens,
        language=language,
        task=task
    )

    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="whisper-ja-{epoch:02d}-{val/wer:.4f}",
            monitor="val/wer",
            mode="min",
            save_top_k=2,
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    # Configure trainer
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": torch.cuda.device_count() if torch.cuda.is_available() else None,
        "precision": precision,
        "callbacks": callbacks,
        "logger": pl.loggers.TensorBoardLogger(output_dir, name="lightning_logs"),
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 10,
        "accumulate_grad_batches": gradient_accumulation,
        "deterministic": False  # Disable deterministic for better performance
    }
    
    # Configure distributed strategy
    if torch.cuda.is_available():
        if deepspeed_stage >= 0:
            trainer_kwargs["strategy"] = DeepSpeedStrategy(config=ds_config_path)
        else:
            # Use DDP if not using DeepSpeed
            trainer_kwargs["strategy"] = "ddp_find_unused_parameters_false"
    
    trainer = pl.Trainer(**trainer_kwargs)

    # Train and evaluate model
    start_time = time.time()
    try:
        # Fit model
        trainer.fit(
            model, 
            data_module, 
            ckpt_path=resume_from_checkpoint
        )
        
        # Test model
        test_results = trainer.test(model, data_module)
        
        # Save model and processor on main process only
        if trainer.is_global_zero:
            model_save_path = os.path.join(output_dir, "model")
            os.makedirs(model_save_path, exist_ok=True)
            model.model.save_pretrained(model_save_path)
            processor.save_pretrained(model_save_path)
            logger.info(f"Model saved to {model_save_path}")
        
        # Return results
        return {
            "status": "success",
            "training_time_hours": (time.time() - start_time) / 3600,
            "output_dir": output_dir,
            "test_results": test_results
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Example usage
    result = train_whisper_model(
        model_id="openai/whisper-small",
        batch_size=4,
        num_workers=4,
        max_epochs=3,
        deepspeed_stage=2,
        precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "16-mixed",
        gradient_accumulation=2
    )
    logger.info(f"Training result: {result}")