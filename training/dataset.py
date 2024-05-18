import random
import json
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple
import en_core_web_sm

class QGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file: str,  # Thay đổi từ csv_file sang json_file
        max_length: int,
        pad_mask_id: int,
        tokenizer: AutoTokenizer
    ) -> None:
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)  # Đọc dữ liệu từ file JSON
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        input_ids, attention_mask = self._encode_text(item["context"])
        labels, _ = self._encode_text(item["question"])
        masked_labels = self._mask_label_padding(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels
        }

    def _encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return (
            encoded_text["input_ids"].squeeze(),
            encoded_text["attention_mask"].squeeze()
        )

    def _mask_label_padding(self, labels: torch.Tensor) -> torch.Tensor:
        labels[labels == self.tokenizer.pad_token_id] = self.pad_mask_id
        return labels
