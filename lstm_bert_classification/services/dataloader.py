import json
from underthesea import word_tokenize
from typing import Mapping, Tuple, List

import torch
from transformers import AutoTokenizer

class Dataset:
    def __init__(
        self,
        json_file: str,
        label_mapping: dict,
        tokenizer: AutoTokenizer,
        word_segment: bool = False,
    ) -> None:
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)["dialogues"]

        self.data = data
        self.label_mapping = label_mapping
        self.sep_token = tokenizer.sep_token
        self.word_segment = word_segment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        dialogue = self.data[index]["sequence"]
        messages = [item["message"] for item in dialogue]
        labels = [item["label"] for item in dialogue]

        if len(messages) != len(labels):
            raise ValueError(f"Length mismatch at index {index}: messages={len(messages)}, labels={len(labels)}")

        if self.word_segment:
            messages = [self._word_segment(msg) for msg in messages]
            
        label_vectors = []
        for label_list in labels:
            label_vector = [0] * len(self.label_mapping)
            for label in label_list if isinstance(label_list, list) else [label_list]:
                if label in self.label_mapping:
                    label_vector[self.label_mapping[label]] = 1
                else:
                    raise ValueError(f"Label '{label}' not found in label_mapping")
            label_vectors.append(label_vector)

        return messages, label_vectors

    def _word_segment(self, text: str) -> str:
        tokens = word_tokenize(text)
        text_segment = " ".join(tokens)
        return text_segment

class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, label_mapping: dict, max_seq_len: int = 10) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = label_mapping
        self.max_seq_len = max_seq_len
        
    def __call__(self, batch: List[Tuple[List[str], List[List[int]]]]) -> Mapping[str, torch.Tensor]:
        all_messages = [item[0] for item in batch]
        all_labels = [item[1] for item in batch]
        
        padded_labels = []
        padded_messages = []
        attention_masks = []
        lengths = []

        for messages, labels in zip(all_messages, all_labels):
            seq_len = len(messages)
            actual_length = min(seq_len, self.max_seq_len)
            lengths.append(actual_length)

            if seq_len < self.max_seq_len:
                padded_msg = messages + [self.tokenizer.pad_token] * (self.max_seq_len - seq_len)
                padded_label = labels + [[0] * len(self.label_mapping)] * (self.max_seq_len - seq_len)
            else:
                padded_msg = messages[:self.max_seq_len]
                padded_label = labels[:self.max_seq_len]

            tokenized = self.tokenizer(
                padded_msg,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                is_split_into_words=False,
            )
            padded_messages.append(tokenized["input_ids"])  # [max_seq_len, max_length]
            attention_masks.append(tokenized["attention_mask"])
            
            label_tensor = torch.tensor(padded_label, dtype=torch.float)
            padded_labels.append(label_tensor)

        input_ids = torch.stack(padded_messages)  # [batch_size, max_seq_len, max_length]
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(padded_labels)  # [batch_size, max_seq_len, num_labels]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "lengths": torch.tensor(lengths, dtype=torch.long)
        }