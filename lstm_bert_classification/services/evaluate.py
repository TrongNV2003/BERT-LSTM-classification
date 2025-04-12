import json
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader, Dataset

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
        self.id2label=id2label

    def evaluate(self):
        self.model.eval()
        results = []
        latencies = []
        all_preds = []
        all_labels = []
        all_masks = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=len(self.test_loader), unit="batches"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                batch_start_time = time.time()
                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_end_time = time.time()
                latency = batch_end_time - batch_start_time
                latencies.append(latency)

                probs = torch.sigmoid(logits)
                batch_preds = (probs > 0.5).float().cpu().numpy()  # [batch_size, seq_len, num_labels]
                batch_labels = labels.cpu().numpy()

                batch_size, seq_len, _ = batch_preds.shape
                valid_mask = np.zeros((batch_size, seq_len), dtype=bool)
                lengths_np = lengths.cpu().numpy()
                for i in range(batch_size):
                    valid_mask[i, :int(lengths_np[i])] = True

                valid_preds = batch_preds[valid_mask]
                valid_labels = batch_labels[valid_mask]

                
                all_preds.append(valid_preds)
                all_labels.append(valid_labels)
                all_masks.append(valid_mask)

                for i in range(batch_size):
                    actual_length = int(lengths_np[i])
                    for j in range(actual_length):
                        true_label_names = self._map_labels(batch_labels[i, j], self.id2label)
                        predicted_label_names = self._map_labels(batch_preds[i, j], self.id2label)
                        results.append({
                            "true_labels": true_label_names,
                            "predicted_labels": predicted_label_names,
                            "latency": float(latency) / (batch_size * actual_length),
                        })

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        num_samples = len(results)
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")

        metrics = {}
        for average_type in ["micro", "macro", "weighted"]:
            avg_metrics = self._calculate_metrics(all_preds, all_labels, average_type)
            metrics[average_type] = avg_metrics
            self._print_metrics(avg_metrics, average_type)

        latency_stats = self._calculate_latency_stats(latencies)
        metrics["latency"] = latency_stats
        print(f"num samples: {num_samples}")
        self._calculate_accuracy(results)


    def _map_labels(self, label_data: list, labels_mapping: dict) -> list:
        return [labels_mapping[idx] for idx, val in enumerate(label_data) if val == 1]


    def _calculate_metrics(self, all_preds: np.ndarray, all_labels: np.ndarray, average_type: str) -> Dict[str, float]:
        metrics = {}
        sample_accuracy = (all_preds == all_labels).all(axis=1).mean()
        metrics["accuracy"] = float(sample_accuracy)
        metrics["precision"] = float(precision_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["recall"] = float(recall_score(all_labels, all_preds, average=average_type, zero_division=0))
        metrics["f1"] = float(f1_score(all_labels, all_preds, average=average_type, zero_division=0))

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


    def _print_metrics(self, metrics: Dict[str, float], average_type: str) -> None:
        print(f"\nMetrics ({average_type}):")
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}")
        print(f"Precision: {metrics['precision'] * 100:.2f}")
        print(f"Recall: {metrics['recall'] * 100:.2f}")
        print(f"F1 Score: {metrics['f1'] * 100:.2f}")


    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        latency_stats = {
            "p95_ms": float(np.percentile(latencies, 95) * 1000),
            "p99_ms": float(np.percentile(latencies, 99) * 1000),
            "mean_ms": float(np.mean(latencies) * 1000),
        }
        print("\nLatency Statistics:")
        print(f"P95 Latency: {latency_stats['p95_ms']:.2f} ms")
        print(f"P99 Latency: {latency_stats['p99_ms']:.2f} ms")
        print(f"Mean Latency: {latency_stats['mean_ms']:.2f} ms")
        
        return latency_stats