import json
import torch
from transformers import T5Tokenizer
from typing import Mapping, Tuple

class QGDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: T5Tokenizer) -> None:
        """
        task:
            - input: article (i.e. context)
            - output: question <sep> answer
        args:
            tokenizer: tokenizer
            data_split: train, validation, test
        """
        
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
        separator = '<sep>'
        
        if question_type == "multiple_choice":
            question = item["question"]
            all_answers = item['options']
            correct_answer_index =  ord(item['answer']) - 65

            curr_correct = all_answers.pop(correct_answer_index)
            target_text = question + ' ' + separator + ' ' + curr_correct
            input_ids, attention_mask = self._encode_text(f"Trắc nghiệm: {context}")
            
        elif question_type == "sentences":
            question = item["question"]
            answer = item["answer"]
            target_text = question + ' ' + separator + ' ' + answer
            input_ids, attention_mask = self._encode_text(f"Tự luận: {context}")

        
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
        
        
class DistractorDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: T5Tokenizer) -> None:
        """
        task:
            - input: question <sep> answer <sep> context
            - output: distractor1 <sep> distractor2 <sep> distractor3
        args:
            tokenizer: tokenizer
            data_split: train, validation, test
        """
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = [item for item in data if item["question_type"] == "multiple_choice"]
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["context"]
        question_type = item["question_type"]
        separator = '<sep>'
        
        if question_type == "multiple_choice":
            question = item["question"]
            all_answers = item['options']
            correct_answer_index =  ord(item['answer']) - 65

            curr_correct = all_answers.pop(correct_answer_index)
            curr_incorrect1 = all_answers[0]
            curr_incorrect2 = all_answers[1]
            curr_incorrect3 = all_answers[2]
            target_text = curr_incorrect1 + ' ' + separator + ' ' + curr_incorrect2 + ' ' + separator + ' ' + curr_incorrect3
            input_text = question + ' ' + separator + ' ' + curr_correct + ' ' + separator + ' ' + context

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