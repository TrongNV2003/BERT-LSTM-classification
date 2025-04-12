import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler, AutoModel

import numpy as np
from tqdm import tqdm
from typing import Optional, Callable
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

from lstm_bert_classification.services.utils import AverageMeter

class BertLSTMModel(nn.Module):
    def __init__(self, bert_model_name: str, num_labels: int, lstm_hidden_size: int = 100, dropout: float = 0.1):
        super(BertLSTMModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,    # 768 dimensions
            hidden_size=lstm_hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, lengths, hidden=None):
        # input_ids: [batch_size, seq_len, max_length]
        batch_size, seq_len, max_length = input_ids.size()

        # Reshape để đưa qua BERT
        flat_input_ids = input_ids.view(-1, max_length)  # [batch_size * seq_len, max_length]
        flat_attention_mask = attention_mask.view(-1, max_length)

        # BERT forward
        bert_outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        
        # Lấy token [CLS] từ kết quả của BERT: [batch_size * seq_len, hidden_size]
        bert_hidden = bert_outputs.last_hidden_state[:, 0, :]

        # Reshape lại để đưa qua LSTM: [batch_size, seq_len, hidden_size]
        lstm_input = bert_hidden.view(batch_size, seq_len, -1)
        pack_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Khởi tạo hidden state nếu chưa được truyền vào
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(lstm_input.device)
            c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(lstm_input.device)
            hidden = (h0, c0)
            
        packed_output, hidden = self.lstm(pack_input, hidden)  # lstm_output: [batch_size, seq_len, lstm_hidden_size]

        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len
        )  # [batch_size, seq_len, lstm_hidden_size]

        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)  # [batch_size, seq_len, num_labels]

        return logits, hidden

class BertRNNModel(nn.Module):
    def __init__(self, bert_model_name: str, num_labels: int, rnn_hidden_size: int = 100, dropout: float = 0.1):
        super(BertRNNModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.rnn = nn.RNN(
            input_size=self.bert.config.hidden_size,  # 768 dimensions
            hidden_size=rnn_hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, lengths, hidden=None):
        # input_ids: [batch_size, seq_len, max_length]
        batch_size, seq_len, max_length = input_ids.size()

        # Reshape để đưa qua BERT
        flat_input_ids = input_ids.view(-1, max_length)  # [batch_size * seq_len, max_length]
        flat_attention_mask = attention_mask.view(-1, max_length)

        # BERT forward
        bert_outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        bert_hidden = bert_outputs.last_hidden_state[:, 0, :]  # Lấy [CLS] token: [batch_size * seq_len, 768]

        # Reshape lại để đưa qua RNN
        rnn_input = bert_hidden.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 768]

        pack_input = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        # RNN forward
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.rnn.hidden_size).to(rnn_input.device)
        packed_output, hidden = self.rnn(pack_input, hidden)  # rnn_output: [batch_size, seq_len, rnn_hidden_size]

        rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,  # Ensure the output is in the same format
            total_length=seq_len
        )
        
        rnn_output = self.dropout(rnn_output)
        logits = self.classifier(rnn_output)  # [batch_size, seq_len, num_labels]

        return logits, hidden

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
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

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

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

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
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = AverageMeter()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)
                    lengths = data["lengths"].to(self.device)

                    logits, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        lengths=lengths,
                    )
                    loss = self.loss_fn(logits, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss.update(loss.item(), input_ids.size(0))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix({"train_loss": train_loss.avg, "lr": current_lr})
                    tepoch.update(1)

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

                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lengths=lengths,
                )
                loss = self.loss_fn(logits, labels)
                eval_loss.update(loss.item(), input_ids.size(0))

                probs = torch.sigmoid(logits)  # [batch_size, seq_len, num_labels]
                preds = (probs > 0.5).float().cpu().numpy()  # [batch_size, seq_len, num_labels]

                # Flatten thành [batch_size * seq_len, num_labels]
                preds_flat = preds.reshape(-1, preds.shape[-1])
                labels_flat = labels.cpu().numpy().reshape(-1, labels.shape[-1])
                    
                all_preds.append(preds_flat)
                all_labels.append(labels_flat)

                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        hamming_score = 1 - hamming_loss(all_labels, all_preds)
        print(f"Hamming Score: {hamming_score * 100:.2f}%")

        accuracy = np.mean((all_preds == all_labels).all(axis=-1))
        self._print_metrics(all_preds, all_labels, "micro")

        return accuracy if self.evaluate_on_accuracy else eval_loss.avg


    def _print_metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> None:
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