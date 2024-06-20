import json
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple
import random

class QGDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: AutoTokenizer, separator = '<sep>') -> None:
        """
        task:
            - input: article (i.e. context)
            - output: question <sep> answer
        args:
            tokenizer: tokenizer
            data_split: train, validation, test
        """
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = [item for item in data if item["question_type"] == "sentences"]
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer
        self.separator = separator
        self.label_mapping = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["context"]

        question = item["question"]
        answer = item["answer"]
        
        target_text = f"{question} {self.separator} {answer}"
        input_ids, attention_mask = self._encode_text(f"{context}")

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
    
    
class MCQ_QGDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: AutoTokenizer, separator = '<sep>') -> None:
        """
        task:
            - input: article (i.e. context)
            - output: question <sep> answer
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
        self.separator = separator
        self.label_mapping = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["context"]
        question = item["question"]
        options = item["options"]
        label_answer =  item["answer"]
        answer = options[self.label_mapping[label_answer]]
            
        target_text = f"{question} {self.separator} {answer}"
        input_ids, attention_mask = self._encode_text(f"{context}")

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
    def __init__(self, json_file: str, max_length: int, pad_mask_id: int, tokenizer: AutoTokenizer, shuffle_distractors=True, separator = '<sep>') -> None:
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
        self.separator = separator
        self.label_mapping = {label: i for i, label in enumerate(["A", "B", "C", "D"])}
        self.all_labels = [0, 1, 2, 3]
        self.shuffle_distractors = shuffle_distractors
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        context = item["context"]
        question = item["question"]
        options = item["options"]
        label_answer = item["answer"]
        
        answer_i = self.label_mapping[label_answer]
        answer = options[answer_i]

        distractor_ids = [i for i in self.all_labels if i != answer_i]
        if self.shuffle_distractors:
            random.shuffle(distractor_ids)
        distractors = [options[i] for i in distractor_ids]
        
        input_text = f"{question} {self.separator} {answer} {self.separator} {context}"
        target_text = f"{distractors[0]}"

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