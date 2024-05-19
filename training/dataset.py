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
        question = item["question"]

        if item['question_type'] == 'multiple_choice':
            # Tiền tố cho câu hỏi trắc nghiệm
            task_prefix = "generate mcq:"
            options = " ".join([f"{chr(65+i)}: {option}" for i, option in enumerate(item['options'])])
            input_text = f"{task_prefix} question: {question} context: {context} options: {options}"
            answer_index = ord(item['answer']) - ord('A')
            answer = item['options'][answer_index]
        else:
            # Tiền tố cho câu hỏi tự luận
            task_prefix = "generate essay:"
            input_text = f"{task_prefix} question: {question} context: {context}"
            answer = item["answer"]

        # Encode the input text and the answer text
        input_ids, attention_mask = self._encode_text(input_text)
        labels, _ = self._encode_text(answer)

        # Apply masking to labels where padding tokens are present
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
