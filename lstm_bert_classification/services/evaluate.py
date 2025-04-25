import json
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader, Dataset

from lstm_bert_classification.utils import constant

class TestingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        model: torch.nn.Module,
        pin_memory: bool,
        test_set: Dataset,
        test_batch_size: int,
        id2label: dict,
        collate_fn: Optional[Callable] = None,
        output_file: Optional[str] = None,
    ) -> None:
        self.output_file = output_file
        self.id2label=id2label
        
        self.device = device
        self.model = model.to(self.device)
        
        self.test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def evaluate(self):
        self.model.eval()
        results, latencies = [], []
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=len(self.test_loader), unit="batches"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                turn_mask = batch["turn_mask"].to(self.device)
                
                batch_size = input_ids.shape[0]
                batch_hidden = None

                start = time.time()
                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lengths=lengths,
                    turn_mask=turn_mask,
                    hidden=batch_hidden,
                )
                latency = time.time() - start
                latencies.append(latency)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5)
                labels_np = labels.cpu().numpy()

                for i in range(batch_size):
                    valid_len = int(lengths[i].item())
                    sample_preds = preds[i, :valid_len]         # [valid_len, num_labels]
                    sample_labels = labels_np[i, :valid_len]    # [valid_len, num_labels]
                    
                    all_preds.append(sample_preds)
                    all_labels.append(sample_labels)
                    
                    for j in range(valid_len):
                        true_labels = self._map_labels(sample_labels[j], self.id2label)
                        predicted_labels = self._map_labels(sample_preds[j], self.id2label)
                        if not predicted_labels:
                            predicted_labels = [self.id2label.get('UNKNOWN', constant.UNKNOWN_LABEL)]
                            
                        results.append({
                            "true_labels": true_labels,
                            "predicted_labels": predicted_labels,
                            "latency": latency / batch_size / valid_len,
                        })

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")

        metrics = {}
        for avg in ["micro", "macro", "weighted"]:
            metrics[avg] = self._calculate_metrics(all_preds, all_labels, avg)
            
        latency_stats = self._calculate_latency_stats(latencies)
        metrics["latency"] = latency_stats
        self._calculate_accuracy(results)
        
        num_samples = len(results)
        print(f"num samples: {num_samples}")
        return metrics

    def _map_labels(self, label_data: list, labels_mapping: dict) -> list:
        return [labels_mapping[idx] for idx, val in enumerate(label_data) if val == 1]

    def _calculate_metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> Dict[str, float]:
        metrics = {}
        metrics["precision"] = float(precision_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["recall"] = float(recall_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["f1"] = float(f1_score(all_labels, all_preds, average=average_type, zero_division=0))
        print(f"\nMetrics ({average_type}):")
        print(f"Precision: {metrics['precision'] * 100:.2f}")
        print(f"Recall: {metrics['recall'] * 100:.2f}")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}")
        return metrics

    def _calculate_accuracy(self, results):
        correct = 0
        correct_one = 0
        total = len(results)
        for item in results:
            true_set = set(item["true_labels"])
            pred_set = set(item["predicted_labels"])
            if true_set == pred_set:
                correct += 1
            if true_set & pred_set:
                correct_one += 1
        accuracy = correct / total if total > 0 else 0
        accuracy_one = correct_one / total if total > 0 else 0
        print(f"\nAccuracy (Match one): {accuracy_one * 100:.2f}%")
        print(f"Accuracy (Match all): {accuracy * 100:.2f}%")


    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        stats = {
            "p95_ms": float(np.percentile(latencies, 95) * 1000),
            "p99_ms": float(np.percentile(latencies, 99) * 1000),
            "mean_ms": float(np.mean(latencies) * 1000),
        }
        print("\nLatency Statistics:")
        print(f"P95 Latency: {stats['p95_ms']:.2f} ms")
        print(f"P99 Latency: {stats['p99_ms']:.2f} ms")
        print(f"Mean Latency: {stats['mean_ms']:.2f} ms")
        return stats