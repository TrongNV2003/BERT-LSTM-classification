import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler

import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Optional, Callable
from sklearn.metrics import precision_score, recall_score, f1_score

from lstm_bert_classification.utils.utils import AverageMeter
from lstm_bert_classification.services.loss import LossFunctionFactory

class TrainingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
        evaluate_on_accuracy: bool = True,
        collator_fn: Optional[Callable] = None,
        focal_loss_gamma: float = 2.0,
        focal_loss_alpha: float = 0.25,
        use_focal_loss: bool = True,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collator_fn,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn,
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.loss_factory = LossFunctionFactory()
        
        if self.use_focal_loss:
            self.loss_fn = self.loss_factory.get_loss("focal_with_pad_mask", gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha)
            logger.info(f"Using Focal Loss with gamma={self.focal_loss_gamma} and alpha={self.focal_loss_alpha}")
        else:
            self.loss_fn = self.loss_factory.get_loss("bce_with_pad_mask")
            logger.info("Using BCE Loss")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        num_training_steps = len(self.train_loader) * epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        self.evaluate_on_accuracy = evaluate_on_accuracy
        self.best_valid_score = 0 if evaluate_on_accuracy else float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_epoch = 0


    def train(self) -> None:
        logger.info(f"Using {'Focal Loss' if self.use_focal_loss else 'BCE Loss'} for training")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = AverageMeter()

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                turn_mask = batch["turn_mask"].to(self.device)
                
                batch_hidden = None
                self.optimizer.zero_grad()
                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lengths=lengths,
                    turn_mask=turn_mask,
                    hidden=batch_hidden
                )
                
                # build mask for padding steps
                pad_mask = turn_mask.unsqueeze(-1).float()
                loss = self.loss_fn(logits, labels, pad_mask)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                train_loss.update(loss.item(), input_ids.size(0))

            valid_score = self._validate(self.valid_loader)
            improved = False

            if self.evaluate_on_accuracy:
                if valid_score > self.best_valid_score + self.early_stopping_threshold:
                    print(f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in val accuracy. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

            else:
                if valid_score < self.best_valid_score - self.early_stopping_threshold:
                    print(f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in validation loss. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

            if improved:
                print(f"Saved best model at epoch {self.best_epoch}.")
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                break

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        all_preds = []
        all_labels = []
        
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)
                lengths = data["lengths"].to(self.device)
                turn_mask = data["turn_mask"].to(self.device)
                
                batch_size = input_ids.size(0)
                batch_hidden = None
                
                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lengths=lengths,
                    hidden=batch_hidden,
                )
            
                # build mask for padding steps
                pad_mask = turn_mask.unsqueeze(-1).float()

                loss = self.loss_fn(logits, labels, pad_mask)
                
                eval_loss.update(loss.item(), input_ids.size(0))

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().cpu().numpy()
                
                for i in range(batch_size):
                    valid_length = lengths[i].item()
                    valid_preds = preds[i][:valid_length]
                    valid_labels_np = labels[i][:valid_length].cpu().numpy()
                    
                    all_preds.append(valid_preds)
                    all_labels.append(valid_labels_np)

                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)

        all_preds = np.vstack([p for p in all_preds if p.size > 0])
        all_labels = np.vstack([l for l in all_labels if l.size > 0])

        accuracy = np.mean((all_preds == all_labels).all(axis=-1))
        self._metrics(all_preds, all_labels, "micro")

        return accuracy if self.evaluate_on_accuracy else eval_loss.avg

    def _metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> None:
        accuracy = np.mean((all_preds == all_labels).all(axis=-1))
        precision = precision_score(all_labels, all_preds, average=average_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=average_type, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=average_type, zero_division=0)
        print(f"\n=== Metrics ({average_type}) ===")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.bert.save_pretrained(self.save_dir)
        torch.save(self.model.state_dict(), f"{self.save_dir}/pytorch_model.bin")