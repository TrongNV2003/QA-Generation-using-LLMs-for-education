import random
import json
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple, List
import en_core_web_sm

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
        self.spacy_tokenizer = en_core_web_sm.load()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        question_type = item.get("question_type", "open_ended")
        
        if question_type == "multiple_choice":
            input_text = f"generate mcq: {item['context']}"
            question = item["question"]
            correct_answer = item["answer"]
            incorrect_answers = item.get("incorrect_answers", [])
            if not incorrect_answers:
                incorrect_answers = self._generate_incorrect_answers(correct_answer)
            answers = [correct_answer] + incorrect_answers
            random.shuffle(answers)
            labels = {
                "question": question,
                "answers": answers,
                "correct_answer": correct_answer
            }
        else:
            input_text = f"generate sentence question: {item['context']}"
            question = item["question"]
            labels = {"question": question}

        input_ids, attention_mask = self._encode_text(input_text)
        masked_labels = self._mask_label_padding(self._encode_text(labels["question"])[0])
        
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

    def _generate_incorrect_answers(self, correct_answer: str) -> List[str]:
        """Generates incorrect answers that are similar to the correct answer."""
        incorrect_answers = []
        for _ in range(3):
            incorrect_answer = self._generate_distractor(correct_answer)
            incorrect_answers.append(incorrect_answer)
        return incorrect_answers

    def _generate_distractor(self, correct_answer: str) -> str:
        """Generates a distractor that is related but incorrect."""
        # Simple example: altering the correct answer slightly to create a distractor
        return correct_answer + random.choice(["ly", "es", "ed", "ing"])

    def shuffle(self, question: str, answer: str) -> Tuple[str, str]:
        shuffled_answer = answer
        while shuffled_answer == answer:
            shuffled_answer = random.choice(self.data)["answer"]
        return question, shuffled_answer

    def corrupt(self, question: str, answer: str) -> Tuple[str, str]:
        doc = self.spacy_tokenizer(question)
        if len(doc.ents) > 1:
            # Replace all entities in the sentence with the same thing
            copy_ent = str(random.choice(doc.ents))
            for ent in doc.ents:
                question = question.replace(str(ent), copy_ent)
        elif len(doc.ents) == 1:
            # Replace the answer with an entity from the question
            answer = str(doc.ents[0])
        else:
            question, answer = self.shuffle(question, answer)
        return question, answer
