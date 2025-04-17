# -*- coding: utf-8 -*-
import argparse
import torch
import sys
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import ( 
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    SchedulerType,
    EarlyStoppingCallback,
    TrainerCallback,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning with AdaLora")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
       "--pure_dataset",
        action="store_true",
        help="Whether or not to use pure dataset for training.",
        required=True,
    )
    parser.add_argument("--language", type=str, help="Language to use for training; e.g., 'Hindi' ", required=False)
    parser.add_argument("--language_abbr", type=str, help="Language to use for training; e.g., 'hi' ", required=False)
    parser.add_argument(
        "--task", type=str, default="transcribe", help="Task to use for training; e.g., 'transcribe' ", required=False
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset to use for training; e.g., 'whisper' ",
        required=False,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset to use for training; e.g., 'whisper' ",
        required=False,
    )
    parser.add_argument(
        "--dataset_in_streaming_mode",
        action="store_true",
        help="Whether to use streaming mode for the dataset.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="lowercase the transcribed text before tokenizing"
    )
    parser.add_argument(
        "--do_remove_punctuation", action="store_true", help="remove punctuation from the transcribed text"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--max_audio_input_length", type=float, default=30.0, help="Maximum audio length in seconds.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=5000,
        help="Number of samples to prefetch in the streaming mode.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        help="Whether or not to pin memory for the DataLoader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=500,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--initialize_from_checkpoint",
        type=str,
        default=None,
        help="If the initialize from a checkpoint folder.",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Whether to use debug mode",
    )
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to save.")
    parser.add_argument("--early_stop_steps", type=int, default=0, help="Whether to use early stop")
    args = parser.parse_args()
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args
def main():
    args = parse_args()
    raw_datasets = load_dataset(args.dataset_path)['train'].shuffle(seed=args.seed).train_test_split(test_size=0.1)
    if not args.pure_dataset:
        raw_datasets = raw_datasets.cast_column("audio",Audio(sampling_rate=16000))
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language="ja", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="ja", task="transcribe")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    raw_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets.column_names["train"], num_proc=2)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.generation_config.language = "ja"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    from spacy import load as spacy_load
    from ginza import set_split_mode as ginza_set_split_mode
    nlp = spacy_load("ja_ginza")
    ginza_set_split_mode(nlp, "C")
    for mod in ['spacy', 'ginza']:
        if mod in sys.modules:
            del sys.modules[mod]
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        normalized_pred_str = [normalizer(pred).strip() for pred in pred_str]
        normalized_label_str = [normalizer(label).strip() for label in label_str]
        pred_str = [" ".join([ str(i) for i in nlp(j) ]) for j in pred_str]
        label_str = [" ".join([ str(i) for i in nlp(j) ]) for j in label_str]
        normalized_pred_str = [" ".join([ str(i) for i in nlp(j) ]) for j in normalized_pred_str]
        normalized_label_str = [" ".join([ str(i) for i in nlp(j) ]) for j in normalized_label_str]
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        normalized_wer = 100 * metric.compute(predictions=normalized_pred_str, references=normalized_label_str)
        return {"wer": wer, "normalized_wer": normalized_wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-ja",  # change to a repo name of your choice
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps= args.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        max_steps=args.max_train_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.checkpointing_steps,
        eval_steps=args.evaluation_steps,
        logging_steps=args.logging_steps,
        report_to=[args.report_to],
        load_best_model_at_end=True,
        metric_for_best_model="normalized_wer",
        greater_is_better=False,
        dataloader_pin_memory=args.dataloader_pin_memory,
        seed=args.seed,
        push_to_hub=False,
        save_safetensors= False,
        optim="adamw_bnb_8bit",
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        eval_strategy = "steps",
    )
    class StopAtStepCallback(TrainerCallback):
        def __init__(self, stop_step):
            self.stop_step = stop_step
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.stop_step:
                control.should_save = True
                return control
        
        def on_save(self, args, state, control, **kwargs):
            if state.global_step >= self.stop_step:
                control.should_training_stop = True
                return control

    stop_step_callback = StopAtStepCallback(args.early_stop_steps) if args.early_stop_steps > 0 else None
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3  # Stop if no improvement after 3 evaluations
    )
    callback_list = [early_stopping_callback, stop_step_callback] if args.early_stop_steps > 0 else [early_stopping_callback]
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callback_list,
    )
    # processor.save_pretrained(training_args.output_dir)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

if __name__ == "__main__":
    main()