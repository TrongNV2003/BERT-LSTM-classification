import json
from underthesea import word_tokenize
from typing import Mapping, Tuple, List, Optional

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
            data = json.load(f)
        self.data = data
        self.label_mapping = label_mapping
        self.sep_token = tokenizer.sep_token
        self.word_segment = word_segment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, list]:
        dialogue = self.data[index]["dialogue"]
        dialogue_id = self.data[index]["id"]
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

        return messages, label_vectors, dialogue_id

    def _word_segment(self, sentence: str) -> str:
        context = word_tokenize(sentence, format="text")
        return context

        
class DataCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        label_mapping: dict,
        max_seq_len: Optional[int] = 6,
        dynamic_padding: bool = False
        ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = label_mapping
        self.max_seq_len = max_seq_len
        self.dynamic_padding = dynamic_padding
        
    def __call__(self, batch: List[Tuple[List[str], List[List[int]], int]]) -> Mapping[str, torch.Tensor]:
        batch_size = len(batch)
        padded_seqs, padded_labs, lengths, dialogue_ids = [], [], [], []

        if self.dynamic_padding:
            max_turns_in_batch = max(len(dialogue[0]) for dialogue in batch)    # dialogue [messages, label_vec, d_id]
        else:
            max_turns_in_batch = self.max_seq_len

        # padding/truncate
        for messages, labels, d_id in batch:
            L = len(messages)
            lengths.append(min(L, max_turns_in_batch))
            dialogue_ids.append(d_id)
            
            if L < max_turns_in_batch:
                padded_seqs.append(messages + [self.tokenizer.pad_token] * (max_turns_in_batch - L))
                padded_labs.append(labels + [[0] * len(self.label_mapping)] * (max_turns_in_batch - L))
            else:
                padded_seqs.append(messages[:max_turns_in_batch])
                padded_labs.append(labels[:max_turns_in_batch])
            
        # flatten for batching tokenization
        flat_texts = [txt for seq in padded_seqs for txt in seq]
        tokenized = self.tokenizer(
            flat_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        n_texts, max_len = tokenized["input_ids"].shape
        
        assert n_texts == batch_size * max_turns_in_batch, \
        f"got {n_texts} texts, expected {batch_size*max_turns_in_batch}"
        
        input_ids = tokenized['input_ids'].view(batch_size, max_turns_in_batch, max_len)
        attention_mask = tokenized['attention_mask'].view(batch_size, max_turns_in_batch, max_len)

        labels_tensor = torch.tensor(padded_labs, dtype=torch.float)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        turn_mask = (torch.arange(max_turns_in_batch)[None, :].to(lengths_tensor.device) < lengths_tensor[:, None])
        dialogue_ids_tensor = torch.tensor(dialogue_ids, dtype=torch.long)
        
        return {
            'input_ids': input_ids,             # [batch_size, seq_len, max_len]
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'lengths': lengths_tensor,
            'turn_mask': turn_mask,
            'dialogue_ids': dialogue_ids_tensor
        }
        
        
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader

#     json_file = "data_speech_final/test.json"
#     unique_labels = ['Tương tác|Đồng ý', 'Tương tác|Chào hỏi', 'Tương tác|UNKNOWN', 'Cung cấp địa điểm giao hàng|Cung cấp vị trí, địa chỉ giao hàng', 'Hỏi địa điểm giao hàng|Hỏi vị trí khách nhận hàng', 'Cung cấp thông tin đơn hàng|Cung cấp tên sản phẩm', 'Hỏi thông tin khi nhận hàng|Hỏi tình trạng nhận hàng', 'Cung cấp thông tin đơn hàng|Cung cấp mã đơn hàng', 'UNKNOWN|UNKNOWN', 'Thông báo trạng thái giao/nhận hàng|Thông báo đã đến nơi', 'Cung cấp thông tin đơn hàng|Cung cấp thời gian giao hàng', 'Cung cấp thông tin đơn hàng|Cung cấp giá sản phẩm', 'Cung cấp hình thức thanh toán đơn hàng|Chuyển khoản', 'Cung cấp thông tin đơn hàng|Cung cấp chi phí đơn hàng', 'Cung cấp địa điểm giao hàng|Mô tả đường đi', 'Hỏi thông tin đơn hàng|Hỏi tên sản phẩm', 'Hỏi thông tin đơn hàng|UNKNOWN', 'Thông báo trạng thái giao/nhận hàng|Nhận hàng được', 'Cung cấp địa điểm giao hàng|Gửi tại nhà và nơi làm việc', 'Tương tác|Yêu cầu nhắc lại thông tin', 'Yêu cầu hỗ trợ|UNKNOWN', 'Cung cấp thông tin đơn hàng|Cung cấp tên khách hàng', 'Tương tác|Cảm thán', 'Khách hàng delay giao|Do không có nhà', 'Cung cấp địa điểm giao hàng|Cung cấp vị trí shipper', 'Cung cấp địa điểm giao hàng|Cung cấp vị trí khách đang đứng', 'Tương tác|Không đồng ý', 'Yêu cầu khi giao hàng|UNKNOWN', 'Hỏi thông tin đơn hàng|Hỏi thời gian giao hàng', 'Cung cấp thông tin đơn hàng|UNKNOWN', 'Tương tác|Đóng góp', 'Cung cấp thông tin người nhận hàng|Giao cho người thân', 'Hỏi thông tin khi nhận hàng|Hỏi người nhận hàng', 'Hỏi thông tin đơn hàng|Hỏi chi phí đơn hàng', 'Cung cấp địa điểm giao hàng|UNKNOWN', 'Hỏi địa điểm giao hàng|Hỏi vị trí shipper', 'Cung cấp địa điểm giao hàng|Cung cấp vị trí Tài xế', 'Thông báo trạng thái giao/nhận hàng|UNKNOWN', 'Yêu cầu khi giao hàng|Gọi trước khi giao', 'Thông báo trạng thái giao/nhận hàng|Không nhận hàng được', 'Cung cấp thông tin đơn hàng|Cung cấp số lượng sản phẩm', 'Cung cấp thông tin người nhận hàng|UNKNOWN', 'Khách hàng delay giao|UNKNOWN', 'Khách hàng huỷ giao hàng|Do khách hàng huỷ đơn', 'Yêu cầu hỗ trợ|Cho xem hàng', 'Khách hàng delay giao|Do đi làm', 'Hỏi địa điểm giao hàng|Hỏi vị trí đặt hàng', 'Cung cấp thông tin đơn hàng|Cung cấp tên shop', 'Tương tác|Không hài lòng', 'Cung cấp địa điểm giao hàng|Đặt trước cửa', 'Hỏi địa điểm giao hàng|Hỏi vị trí tài xế', 'Cung cấp thông tin người nhận hàng|Giao cho hàng xóm', 'Cung cấp thông tin người nhận hàng|Giao cho bảo vệ', 'Hướng dẫn|UNKNOWN', 'Cung cấp thông tin đơn hàng|Cung cấp tên sàn thương mại điện tử', 'Khách hàng huỷ giao hàng|UNKNOWN', 'Khách hàng yêu cầu trả hàng|UNKNOWN', 'Cung cấp hình thức thanh toán đơn hàng|UNKNOWN', 'Cung cấp thông tin đơn hàng|Cung cấp tên đơn vị vận chuyển', 'Hỏi địa điểm giao hàng|UNKNOWN', 'Cung cấp chính sách|UNKNOWN', 'Hỏi thông tin khi nhận hàng|UNKNOWN', 'Khách hàng yêu cầu trả hàng|Do hàng khác với mô tả', 'Shipper delay giao|Do gọi khách nhiều lần nhưng không bắt máy', 'Shipper delay giao|Do khách hàng từ chối nhận', 'Báo cáo chất lượng hàng hoá|Hàng khác với mô tả', 'Tài xế delay giao|UNKNOWN', 'Cung cấp thông tin đơn hàng|Cung cấp số điện thoại khách hàng', 'Yêu cầu hỗ trợ|Cho thử hàng', 'Hỏi thông tin đơn hàng|Hỏi tên shop', 'Hỏi thông tin đơn hàng|Hỏi số lượng sản phẩm', 'Cung cấp thông tin người nhận hàng|Giao cho chính khách hàng', 'Yêu cầu hỗ trợ|Hỗ trợ đổi trả hàng tại chỗ', 'Tài xế delay giao|Do gọi khách nhiều lần nhưng không bắt máy', 'Cung cấp địa điểm giao hàng|Cung cấp vị trí tài xế', 'Hỏi địa điểm giao hàng|Hỏi vị trí Tài xế', 'Cung cấp thông tin đơn hàng|Cung cấp kích thước sản phẩm', 'Cung cấp chính sách|Cung cấp chính sách nhận hàng', 'Shipper delay giao|Do khách hàng không có nhà', 'Cung cấp hình thức thanh toán đơn hàng|Thanh toán trả sau', 'Hỏi thông tin khi nhận hàng|Hỏi hình thức thanh toán', 'Yêu cầu hỗ trợ|Đồng kiểm ngoại quan', 'Yêu cầu hỗ trợ|Hỗ trợ hình ảnh, video', 'Cung cấp hình thức thanh toán đơn hàng|Bằng tiền mặt', 'Tài xế delay giao|Do khách hàng từ chối nhận', 'Yêu cầu hỗ trợ|Hỗ trợ bê vác', 'Hỏi thông tin đơn hàng|Cung cấp chi phí đơn hàng', 'Yêu cầu hỗ trợ|Nhận hàng một phần', 'Yêu cầu khi giao hàng|Nhắn tin trước khi giao', 'Giục giao/trả hàng|Giục giao hàng', 'Cung cấp địa điểm giao hàng|Gửi vào kho', 'Tài xế delay giao|Do khách hàng không có nhà', 'Yêu cầu hỗ trợ|Đồng kiểm chi tiết', 'Báo cáo chất lượng hàng hoá|Nhầm hàng', 'Báo cáo chất lượng hàng hoá|UNKNOWN', 'Tương tác|Hài lòng', 'Hướng dẫn|Hướng dẫn gửi hàng', 'Hỏi thông tin đơn hàng|Hỏi tên đơn vị vận chuyển', 'Shipper delay giao|UNKNOWN', 'Hướng dẫn|Hướng dẫn quy trình đổi trả sản phẩm', 'Yêu cầu khi giao hàng|Đổi size', 'Cung cấp hình thức thanh toán đơn hàng|Thanh toán trả trước', 'Khách hàng delay giao|Do về quê', 'Hướng dẫn|Hướng dẫn thao tác app', 'Hỏi thông tin đơn hàng|Hỏi kích thước sản phẩm', 'Khách hàng huỷ giao hàng|Do trùng đơn hàng', 'Khách hàng yêu cầu trả hàng|Do sai đơn hàng', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi quy trình đổi trả sản phẩm', 'Báo cáo chất lượng hàng hoá|Hàng hỏng vỡ', 'Giục giao/trả hàng|UNKNOWN', 'Khách hàng yêu cầu trả hàng|Yêu cầu hoàn tiền', 'Hỏi thông tin đơn hàng|Hỏi tên sàn thương mại điện tử', 'Khách hàng huỷ giao hàng|Do shop huỷ đơn hàng', 'Báo cáo chất lượng hàng hoá|Thiếu hàng', 'Hỏi thông tin đơn hàng|Hỏi nguồn gốc sản phẩm', 'Hỏi đánh giá, chính sách hậu nhận hàng|UNKNOWN', 'Yêu cầu khi giao hàng|Bấm chuông', 'Cung cấp thông tin đơn hàng|Cung cấp cân nặng sản phẩm', 'Cung cấp chính sách|Cung cấp chương trình giảm giá, khuyến mại', 'Khách hàng huỷ giao hàng|Do giao sai đơn', 'Khách hàng huỷ giao hàng|Do khách hàng nhận hàng tại bưu cục', 'Cung cấp chính sách|Cung cấp chính sách hoàn tiền', 'Yêu cầu khi giao hàng|Đổi mẫu mã', 'Hỏi thông tin đơn hàng|Hỏi cân nặng sản phẩm', 'Cung cấp thông tin người nhận hàng|Giao cho lễ tân', 'Yêu cầu khi giao hàng|Đổi màu', 'Khách hàng yêu cầu trả hàng|Do hàng lỗi', 'Cung cấp thông tin đơn hàng|Cung cấp thương hiệu sản phẩm', 'Báo cáo chất lượng hàng hoá|Sai cân', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi chính sách nhận hàng', 'Hỏi thông tin đơn hàng|Hỏi SĐT shop', 'Yêu cầu hỗ trợ|Nhận hàng toàn bộ', 'Cung cấp hình thức thanh toán đơn hàng|Quét mã QR', 'Yêu cầu hỗ trợ|Đồng kiểm mã imei', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi chương trình giảm giá, khuyến mại', 'Hướng dẫn|Hướng dẫn chia sẻ định vị', 'Hỏi thông tin đơn hàng|Hỏi tên thương hiệu sản phẩm', 'Shipper delay giao|Do gặp sự cố', 'Hướng dẫn|Hướng dẫn quy trình huỷ đơn hàng', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi quy trình huỷ đơn hàng', 'Hướng dẫn|Hướng dẫn thanh toán', 'Tài xế delay giao|Do shop huỷ đơn hàng', 'Cung cấp chính sách|Cung cấp chính sách bảo hành sản phẩm', 'Tài xế delay giao|Do lạc đường', 'Khách hàng yêu cầu trả hàng|Do hàng hỏng vỡ', 'Hỏi thông tin đơn hàng|Hỏi số điện thoại shop', 'Cung cấp địa điểm giao hàng|Gửi tủ locker', 'Giục giao/trả hàng|Giục trả hàng', 'Yêu cầu hỗ trợ|Đồng kiểm chứng từ', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi chính sách hoàn tiền', 'Cung cấp địa điểm giao hàng|Gửi vào hộp thư', 'Yêu cầu khi giao hàng|Không bấm chuông', 'Cung cấp hình thức thanh toán đơn hàng|Thanh toán qua ví', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi chính sách bảo hành sản phẩm', 'Yêu cầu hỗ trợ|Hỗ trợ lắp đặt', 'Yêu cầu khi giao hàng|Gõ cửa', 'Báo cáo chất lượng hàng hoá|Hàng thất lạc', 'Cung cấp hình thức thanh toán đơn hàng|Thanh toán bằng thẻ tín dụng', 'Shipper delay giao|Do shop huỷ đơn hàng', 'Tài xế delay giao|Do tắc đường', 'Yêu cầu hỗ trợ|Đồng kiểm date', 'Báo cáo chất lượng hàng hoá|Ướt hàng', 'Tài xế delay giao|Do gặp sự cố', 'Hỏi đánh giá, chính sách hậu nhận hàng|Hỏi đánh giá chất lượng sản phẩm', 'Khách hàng delay giao|Do đi du lịch', 'Shipper delay giao|Do lạc đường']

#     label2id = {label: idx for idx, label in enumerate(unique_labels)}
#     id2label = {idx: label for idx, label in enumerate(unique_labels)}
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=True)
    
#     dataset = Dataset(json_file, label2id, tokenizer, word_segment=True)
#     for i in range(len(dataset)):
#         messages, labels, dialogue_id = dataset[1]
#         print(f"Dialogue ID: {dialogue_id}")
#         print("Messages:", messages)
#         break
    
#     collator = DataCollator(
#         tokenizer,
#         max_length=256,
#         label_mapping=label2id,
#         max_seq_len=6,
#         dynamic_padding=True,
#     )
    
#     train_loader = DataLoader(
#             dataset,
#             batch_size=3,
#             num_workers=1,
#             pin_memory=False,
#             shuffle=False,
#             collate_fn=collator,
#         )
#     for batch in train_loader:
#         print("Input IDs:", batch['input_ids'])
#         print("Attention Mask:", batch['attention_mask'])
#         print("Labels:", batch['labels'])
#         print("Lengths:", batch['lengths'])
#         print("Turn Mask:", batch['turn_mask'])
#         print("Dialogue IDs:", batch['dialogue_ids'])
#         break
