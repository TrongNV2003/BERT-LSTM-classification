import time
import random
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer

from multi_intent_classification.services.evaluate import TestingArguments
from multi_intent_classification.services.dataloader import Dataset, LlmDataCollator
from multi_intent_classification.services.trainer import TrainingArguments, BertRNNModel, BertLSTMModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_vram_usage(device):
    """Trả về VRAM tối đa đã sử dụng trong quá trình chạy (GB)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)

def count_parameters(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2, required=True)
parser.add_argument("--device", type=str, default="cuda", required=True)
parser.add_argument("--seed", type=int, default=42, required=True)
parser.add_argument("--epochs", type=int, default=10, required=True)
parser.add_argument("--learning_rate", type=float, default=3e-5, required=True)
parser.add_argument("--weight_decay", type=float, default=0.01, required=True)
parser.add_argument("--warmup_steps", type=int, default=50, required=True)
parser.add_argument("--max_length", type=int, default=256, required=True)
parser.add_argument("--pad_mask_id", type=int, default=-100, required=True)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True)
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--train_batch_size", type=int, default=16, required=True)
parser.add_argument("--val_batch_size", type=int, default=8, required=True)
parser.add_argument("--test_batch_size", type=int, default=8, required=True)
parser.add_argument("--train_file", type=str, default="dataset/train.json", required=True)
parser.add_argument("--val_file", type=str, default="dataset/val.json", required=True)
parser.add_argument("--test_file", type=str, default="dataset/test.json", required=True)
parser.add_argument("--output_dir", type=str, default="./models/classification", required=True)
parser.add_argument("--record_output_file", type=str, default="output.json")
parser.add_argument("--early_stopping_patience", type=int, default=3, required=True)
parser.add_argument("--early_stopping_threshold", type=float, default=0.001, required=True)
parser.add_argument("--evaluate_on_accuracy", action="store_true", default=False)
args = parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer

def get_model(
    checkpoint: str, device: str, num_labels: str):
    # model = BertRNNModel(bert_model_name=checkpoint, num_labels=num_labels)
    model = BertLSTMModel(bert_model_name=checkpoint, num_labels=num_labels)
    return model.to(device)

if __name__ == "__main__":
    set_seed(args.seed)

    unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    tokenizer = get_tokenizer(args.model)
    
    train_set = Dataset(json_file=args.train_file, label_mapping=label2id, tokenizer=tokenizer)
    val_set = Dataset(json_file=args.val_file, label_mapping=label2id, tokenizer=tokenizer)
    test_set = Dataset(json_file=args.test_file, label_mapping=label2id, tokenizer=tokenizer)

    collator = LlmDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_mapping=label2id,
        max_seq_len=8,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, args.device, num_labels=len(unique_labels))
    
    count_parameters(model)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model_name = args.model.split('/')[-1]
    save_dir = f"{args.output_dir}-{model_name}"

    start_time = time.time()
    trainer = TrainingArguments(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=val_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.val_batch_size,
        collator_fn=collator,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        evaluate_on_accuracy=args.evaluate_on_accuracy,
    )
    trainer.train()

    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60} mins")

    if torch.cuda.is_available():
        max_vram = get_vram_usage(device)
        print(f"VRAM tối đa tiêu tốn khi huấn luyện: {max_vram:.2f} GB")


    # Test model
    # tuned_model = BertRNNModel(f"{save_dir}", num_labels=len(unique_labels)).to(args.device)
    tuned_model = BertLSTMModel(f"{save_dir}", num_labels=len(unique_labels)).to(args.device)
    tuned_model.load_state_dict(torch.load(f"{save_dir}/pytorch_model.bin"))
    tester = TestingArguments(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        model=tuned_model,
        pin_memory=args.pin_memory,
        test_set=test_set,
        test_batch_size=args.test_batch_size,
        id2label=id2label,
        collate_fn=collator,
        output_file=args.record_output_file,
    )
    tester.evaluate()

    print(f"\nmodel: {args.model}")
    print(f"params: lr {args.learning_rate}, epoch {args.epochs}")

