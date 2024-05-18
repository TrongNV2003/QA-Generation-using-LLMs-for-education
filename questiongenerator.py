import json
import numpy as np
import random
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from typing import Any, List, Mapping, Tuple


class QuestionGenerator:
    """A transformer-based NLP system for generating reading comprehension-style questions from
    texts. It can generate full sentence questions, multiple choice questions, or a mix of the
    two styles.
    """

    def __init__(self) -> None:
        QG_PRETRAINED = "t5-base-question-generator"
        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.qg_tokenizer = AutoTokenizer.from_pretrained(
            QG_PRETRAINED, use_fast=False)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

    def generate(
        self,
        article: str,
        num_questions: int = None,
        answer_style: str = "all"
    ) -> List:
        """Takes an article and generates a set of question and answer pairs.
        answer_style should be selected from ["all", "sentences", "multiple_choice"].
        """

        print("Generating questions...\n")

        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions), len(qg_answers)
        )
        assert len(generated_questions) == len(qg_answers), message

        qa_list = self._get_all_qa_pairs(generated_questions, qg_answers, num_questions)

        return qa_list

    def generate_qg_inputs(self, text: str, answer_style: str) -> Tuple[List[str], List[str]]:
        """Given a text, returns a list of model inputs and a list of corresponding answers."""

        VALID_ANSWER_STYLES = ["all", "sentences", "multiple_choice"]

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(
                "Invalid answer style {}. Please choose from {}".format(
                    answer_style, VALID_ANSWER_STYLES
                )
            )

        inputs = []
        answers = []

        if answer_style == "sentences" or answer_style == "all":
            segments = self._split_into_segments(text)

            for segment in segments:
                sentences = self._split_text(segment)
                prepped_inputs, prepped_answers = self._prepare_qg_inputs(
                    sentences, segment, answer_style
                )
                inputs.extend(prepped_inputs)
                answers.extend(prepped_answers)

        if answer_style == "multiple_choice" or answer_style == "all":
            sentences = self._split_text(text)
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(
                sentences
            )
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs: List) -> List[str]:
        """Given a list of concatenated answers and contexts, generates a list of questions."""
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text: str) -> List[str]:
        """Splits the text into sentences, and attempts to split or truncate long sentences."""
        MAX_SENTENCE_LEN = 128
        sentences = re.findall(".*?[.!\?]", text)
        cut_sentences = []

        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                cut_sentences.extend(re.split("[,;:)]", sentence))

        # remove useless post-quote sentence fragments
        cut_sentences = [s for s in sentences if len(s.split(" ")) > 5]
        sentences = sentences + cut_sentences

        return list(set([s.strip(" ") for s in sentences]))

    def _split_into_segments(self, text: str) -> List[str]:
        """Splits a long text into segments short enough to be input into the transformer network.
        Segments are used as context for question generation.
        """
        MAX_TOKENS = 512
        paragraphs = text.split("\n")
        tokenized_paragraphs = [
            self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0
        ]
        segments = []

        while len(tokenized_paragraphs) > 0:
            segment = []

            while len(segment) < MAX_TOKENS and len(tokenized_paragraphs) > 0:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

    def _prepare_qg_inputs(self, sentences: List[str], text: str, answer_style: str) -> Tuple[List[str], List[str]]:
        """Uses extracted entities or key phrases as answers and the text as context.
        Returns a tuple of (model inputs, answers).
        """
        inputs = []
        answers = []

        for sentence in sentences:
            if answer_style == "sentences":
                qg_input = f"generate sentence question: {sentence} {text}"
            elif answer_style == "multiple_choice":
                qg_input = f"generate mcq: {sentence} {text}"
            else:
                qg_input = f"{sentence} {text}"
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    def _prepare_qg_inputs_MC(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """Prepares inputs for multiple-choice questions"""
        inputs_from_text = []
        answers_from_text = []

        for sentence in sentences:
            options = self._generate_mcq_options(sentence)
            if options:
                qg_input = f"generate mcq: {sentence} {options['context']}"
                inputs_from_text.append(qg_input)
                answers_from_text.append(options)

        return inputs_from_text, answers_from_text

    def _generate_mcq_options(self, sentence: str) -> Mapping[str, Any]:
        """Generates multiple-choice options for a given sentence."""
        # Extracting a correct answer from the sentence
        correct_answer = sentence.split()[-1]  # Just as an example, picking the last word as the correct answer

        # Generating distractors (incorrect options)
        distractors = [self._generate_distractor(correct_answer) for _ in range(3)]

        context = " ".join(sentence.split())
        return {
            "context": context,
            "options": distractors + [correct_answer],
            "correct": correct_answer
        }

    def _generate_distractor(self, correct_answer: str) -> str:
        """Generates a distractor that is related but incorrect."""
        # Simple example: altering the correct answer slightly to create a distractor
        return correct_answer + random.choice(["ly", "es", "ed", "ing"])

    @torch.no_grad()
    def _generate_question(self, qg_input: str) -> str:
        """Takes qg_input which is the concatenated answer and context, and uses it to generate
        a question sentence. The generated question is decoded and then returned.
        """
        encoded_input = self._encode_qg_input(qg_input)
        output = self.qg_model.generate(input_ids=encoded_input["input_ids"], max_new_tokens=30)
        question = self.qg_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return question

    def _encode_qg_input(self, qg_input: str) -> torch.tensor:
        """Tokenizes a string and returns a tensor of input ids corresponding to indices of tokens in
        the vocab.
        """
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_all_qa_pairs(self, generated_questions: List[str], qg_answers: List[str], num_questions: int = None):
        """Formats question and answer pairs without ranking or filtering."""
        qa_list = []

        for question, answer in zip(generated_questions, qg_answers):
            qa = {
                "question": question.split("?")[0] + "?",
                "answer": answer
            }
            qa_list.append(qa)

        if num_questions:
            qa_list = qa_list[:num_questions]

        return qa_list


def print_qa(qa_list: List[Mapping[str, str]], show_answers: bool = True) -> None:
    """Formats and prints a list of generated questions and answers."""

    for i in range(len(qa_list)):
        # wider space for 2 digit q nums
        space = " " * int(np.where(i < 9, 3, 4))

        print(f"{i + 1}) Q: {qa_list[i]['question']}")

        answer = qa_list[i]["answer"]
        
        # Check if the answer is a dictionary (MCQ format)
        if isinstance(answer, dict) and 'options' in answer and 'correct' in answer:
            print(f"{space}Context: {answer['context']}")
            for idx, option in enumerate(answer['options']):
                print(f"{space}{idx + 1}. {option}")
            if show_answers:
                correct_option_idx = answer['options'].index(answer['correct']) + 1
                print(f"{space}Correct Answer: {correct_option_idx}. {answer['correct']}\n")
        else:
            if show_answers:
                print(f"{space}A: {answer}\n")


def save_qa_to_txt(qa_list: List[Mapping[str, str]], file_path: str) -> None:
    """Saves a list of generated questions and answers to a text file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for i in range(len(qa_list)):
            space = " " * int(np.where(i < 9, 3, 4))
            file.write(f"{i + 1}) Q: {qa_list[i]['question']}\n")

            answer = qa_list[i]["answer"]

            # Check if the answer is a dictionary (MCQ format)
            if isinstance(answer, dict) and 'options' in answer and 'correct' in answer:
                file.write(f"{space}Context: {answer['context']}\n")
                for idx, option in enumerate(answer['options']):
                    file.write(f"{space}{idx + 1}. {option}\n")
                correct_option_idx = answer['options'].index(answer['correct']) + 1
                file.write(f"{space}Correct Answer: {correct_option_idx}. {answer['correct']}\n")
            else:
                file.write(f"{space}A: {answer}\n")

            file.write("\n")