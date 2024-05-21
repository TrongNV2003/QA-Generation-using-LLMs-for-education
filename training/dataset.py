import json
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple

class QGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file: str,
        max_length: int,
        pad_mask_id: int,
        tokenizer: AutoTokenizer
    ) -> None:
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["context"]
        question_type = item["question_type"]
        
        
        if question_type == "multiple_choice":
            question = "trắc nghiệm: " + item["question"]
            options = " ".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(item["options"])])
            answer_index = ord(item["answer"]) - 65
            answer = item["options"][answer_index]
            target_text = f"{question} {options} Trả lời: {answer}"
            
        elif question_type == "sentences":
            question = "tự luận: " + item["question"]
            answer = item["answer"]
            target_text = f"{question} Trả lời: {answer}"
        
        input_text = f"Context: {context}"

        input_ids, attention_mask = self._encode_text(input_text)
        labels, _ = self._encode_text(target_text)
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