import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


file_name = "output.json"
with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if not item["predicted_labels"]:
        item["predicted_labels"] = ["UNKNOWN"]

def calculate_accuracy(results):
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

all_labels = set()
for item in data:
    all_labels.update(item["true_labels"])
    all_labels.update(item["predicted_labels"])
all_labels = sorted(list(all_labels))

true_multi_hot = []
pred_multi_hot = []

for item in data:
    true_vector = [1 if label in item["true_labels"] else 0 for label in all_labels]
    pred_vector = [1 if label in item["predicted_labels"] else 0 for label in all_labels]
    true_multi_hot.append(true_vector)
    pred_multi_hot.append(pred_vector)

true_multi_hot = np.array(true_multi_hot)
pred_multi_hot = np.array(pred_multi_hot)

metrics = {}
for average_type in ["micro", "macro", "weighted"]:
    precision = precision_score(true_multi_hot, pred_multi_hot, average=average_type, zero_division=0)
    recall = recall_score(true_multi_hot, pred_multi_hot, average=average_type, zero_division=0)
    f1 = f1_score(true_multi_hot, pred_multi_hot, average=average_type, zero_division=0)
    metrics[average_type] = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
print(f"Evaluation for {file_name}:")
print("=== Evaluation Metrics ===")
calculate_accuracy(data)
for avg_type, scores in metrics.items():
    print(f"\nMetrics ({avg_type}):")
    print(f"Precision: {scores['precision'] * 100:.2f}%")
    print(f"Recall: {scores['recall'] * 100:.2f}%")
    print(f"F1 Score: {scores['f1'] * 100:.2f}%")


# # (Tùy chọn) In chi tiết từng nhãn
# print("\n=== Per-label Metrics ===")
# for i, label in enumerate(all_labels):
#     true_col = true_multi_hot[:, i]
#     pred_col = pred_multi_hot[:, i]
#     if np.sum(true_col) > 0 or np.sum(pred_col) > 0:  # Chỉ in nhãn có giá trị
#         p = precision_score(true_col, pred_col, zero_division=0)
#         r = recall_score(true_col, pred_col, zero_division=0)
#         f = f1_score(true_col, pred_col, zero_division=0)
#         print(f"{label}:")
#         print(f"  Precision: {p * 100:.2f}%")
#         print(f"  Recall: {r * 100:.2f}%")
#         print(f"  F1: {f * 100:.2f}%")