import json
from collections import defaultdict

def group_dialogues(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues_dict = defaultdict(list)

    for item in data:
        dialogue_id = item["id"]
        message = item["current_message"]
        label_intent = item["label_intent"]
        
        dialogues_dict[dialogue_id].append({
            "message": message,
            "label": label_intent
        })

    grouped_data = {
        "dialogues": [
            {
                "id": dialogue_id,
                "sequence": sequence
            }
            for dialogue_id, sequence in dialogues_dict.items()
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, ensure_ascii=False, indent=4)

    print(f"Đã nhóm dữ liệu từ {input_file} và lưu vào {output_file}")
    print(f"Tổng số hội thoại: {len(grouped_data['dialogues'])}")

if __name__ == "__main__":
    input_files = {
        "train": "dataset/train.json",
        "val": "dataset/val.json",
        "test": "dataset/test.json"
    }
    output_files = {
        "train": "dialogue_dataset/train.json",
        "val": "dialogue_dataset/val.json",
        "test": "dialogue_dataset/test.json"
    }

    for split in ["train", "val", "test"]:
        group_dialogues(input_files[split], output_files[split])