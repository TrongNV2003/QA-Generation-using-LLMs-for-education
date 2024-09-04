import numpy as np
import random
import torch
import re
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Any, List, Mapping, Tuple

class QuestionAnswerGenerator:
    """A transformer-based NLP system for generating reading comprehension-style questions from texts."""

    def __init__(self) -> None:
        self.SEQ_LENGTH = 512
        QG_PRETRAINED = "Trongdz/vi-T5-QA-generation-for-philosophy"
        self.qg_tokenizer = T5Tokenizer.from_pretrained(QG_PRETRAINED)
        self.qg_tokenizer.add_special_tokens({"sep_token": "<sep>"})
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED, torch_dtype=torch.bfloat16)
        self.qg_model.to(self.device)
        self.qg_model.eval()
        
        MCQ_QG_PRETRAINED = "Trongdz/vi-T5-QA-generation-MCQ-for-philosophy"
        self.mcq_qg_model = AutoModelForSeq2SeqLM.from_pretrained(MCQ_QG_PRETRAINED, torch_dtype=torch.bfloat16)
        self.mcq_qg_model.to(self.device)
        self.mcq_qg_model.eval()
        
        QD_PRETRAINED = "Trongdz/vi-T5-QA-Distractor-in-MCQ-for-philosophy"
        self.qd_model = AutoModelForSeq2SeqLM.from_pretrained(QD_PRETRAINED, torch_dtype=torch.bfloat16)
        self.qd_model.to(self.device)
        self.qd_model.eval()

    def generate(self, context: str, num_questions: int = 5, answer_style: str = "sentences") -> List:
        """Takes a context and generates a set of question and answer pairs."""

        print("Generating questions...\n")

        inputs, questions, answers = self.generate_qa_from_inputs(context, answer_style, num_questions)
        qa_list = self._get_all_qa_pairs(questions, answers, context, answer_style)

        return qa_list
    
    def generate_qa_from_inputs(self, context: str, answer_style: str, num_questions: int) -> Tuple[List[str], List[str], List[str]]:
        """Given a text, returns a list of model inputs, questions, and answers."""

        VALID_ANSWER_STYLES = ["sentences", "multiple_choice"]

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(
                "Invalid answer style {}. Please choose from {}".format(answer_style, VALID_ANSWER_STYLES)
            )
        
        if answer_style == "sentences":
            segments = self._split_context(context)
            for segment in segments:
                sentences = self._split_into_sentences(segment)
                inputs, questions, answers = self._generate_qa(sentences)
            
        elif answer_style == "multiple_choice":
            sentences = self._split_into_sentences(context)
            inputs, questions, answers = self._generate_qa_mcq(sentences)

        # if len(questions) > num_questions:
        #     questions, answers = questions[:num_questions], answers[:num_questions]

        # return inputs, questions, answers
    
        question_lengths = [len(q.split()) for q in questions]
        sorted_indices = np.argsort(question_lengths)[::-1]  # Sort questions by length in descending order

        sorted_questions = [questions[i] for i in sorted_indices]
        sorted_answers = [answers[i] for i in sorted_indices]

        if len(sorted_questions) > num_questions:
            sorted_questions, sorted_answers = sorted_questions[:num_questions], sorted_answers[:num_questions]

        return inputs, sorted_questions, sorted_answers
    
    def _split_context(self, text: str) -> List[str]:
        """Splits a long text into segments short enough to be input into the transformer network.
        Segments are used as context for question generation.
        """
        MAX_TOKENS = 512
        paragraphs = text.split("\n")
        tokenized_paragraphs = [self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0]
        segments = []

        while tokenized_paragraphs:
            segment = []

            while len(segment) < MAX_TOKENS and tokenized_paragraphs:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]
    
    def _split_into_sentences(self, text: str) -> List[str]:
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


    @torch.no_grad()
    def _generate_qa(self, context: List[str]) -> Tuple[List[str], List[str], List[str]]:
        inputs, questions, answers = [], [], []
        for sentences in context:
            qg_input = f"{sentences}"
            encoded_input = self._encode_qg_input(qg_input)
            outputs = self.qg_model.generate(
                input_ids=encoded_input["input_ids"], 
                max_new_tokens=128, 
                num_beams=5,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                do_sample=True
            )
            for output in outputs:
                correct_answer = self.qg_tokenizer.decode(output, skip_special_tokens=False)
                correct_answer = correct_answer.replace(self.qg_tokenizer.pad_token, "").replace(self.qg_tokenizer.eos_token, "")
                question_answer_split = correct_answer.split(self.qg_tokenizer.sep_token)
                if len(question_answer_split) == 2:
                    # valid Question + Answer output
                    question, answer = question_answer_split[0].strip(), question_answer_split[1].strip()
                    inputs.append(qg_input)
                    questions.append(question)
                    answers.append(answer)

        return inputs, questions, answers

    @torch.no_grad()
    def _generate_qa_mcq(self, context: List[str]) -> Tuple[List[str], List[str], List[str]]:
        inputs, questions, answers = [], [], []
        for sentences in context:
            qg_input = f"{sentences}"
            encoded_input = self._encode_qg_input(qg_input)
            outputs = self.mcq_qg_model.generate(
                input_ids=encoded_input["input_ids"], 
                max_new_tokens=128,
                num_beams=5,
                no_repeat_ngram_size=2,
                do_sample=True
            )
            for output in outputs:
                correct_answer = self.qg_tokenizer.decode(output, skip_special_tokens=False)
                correct_answer = correct_answer.replace(self.qg_tokenizer.pad_token, "").replace(self.qg_tokenizer.eos_token, "")
                question_answer_split = correct_answer.split(self.qg_tokenizer.sep_token)
                if len(question_answer_split) == 2:

                    question, answer = question_answer_split[0].strip(), question_answer_split[1].strip()
                    inputs.append(qg_input)
                    questions.append(question)
                    answers.append(answer)

        return inputs, questions, answers
    
    @torch.no_grad()
    def _generate_distractors(self, question: str, correct_answer: str, context: str) -> List[str]:
        distractors = set()
        attempts = 0
        for sentences in context:
            distractor_input = f"{question} {self.qg_tokenizer.sep_token} {correct_answer} {self.qg_tokenizer.sep_token} {sentences}"

            while len(distractors) < 3 and attempts < 10:
                attempts += 1
                input_ids = self._encode_qg_input(distractor_input)
                outputs = self.qd_model.generate(
                    input_ids=input_ids["input_ids"], 
                    max_new_tokens=128, 
                    temperature=0.9,
                    do_sample=True
                    )
                for output in outputs:
                    generated_distractor = self.qg_tokenizer.decode(output, skip_special_tokens=True)
                    generated_distractor = generated_distractor.replace(self.qg_tokenizer.pad_token, "").replace(self.qg_tokenizer.eos_token, "")
                    generated_distractor = generated_distractor.split(self.qg_tokenizer.sep_token)
                    
                    for distractor in generated_distractor:
                        if distractor.lower() != correct_answer.lower() and distractor not in distractors:
                            distractors.add(distractor)

        return list(distractors)

    def _encode_qg_input(self, qg_input: str) -> torch.Tensor:
        """Tokenizes a string and returns a tensor of input ids."""
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _get_all_qa_pairs(self, questions: List[str], answers: List[str], context: str, answer_style: str) -> List[Mapping[str, str]]:
        """Formats question and answer pairs"""
        if answer_style == "multiple_choice":
            qa_list = []
            for question, answer in zip(questions, answers):
                distractors = self._generate_distractors(question, answer, context)
                
                options = [answer] + distractors[:3]
                random.shuffle(options)
                qa_list.append({
                    "question": question.split("?")[0] + "?",
                    "answer": {
                        "options": options,
                        "correct": answer,
                    }
                })
        else:
            qa_list = [{"question": question.split("?")[0] + "?", "answer": answer} for question, answer in zip(questions, answers)]
        return qa_list


def print_qa(qa_list: List[Mapping[str, str]], show_answers: bool = True) -> None:
    """Formats and prints a list of generated questions and answers."""
    for i, qa in enumerate(qa_list):
        space = " " * (3 if i < 9 else 4)
        print(f"{i + 1}) Question: {qa['question']}")
        answer = qa["answer"]
        if isinstance(answer, dict):  # multiple choice format
            for idx, option in enumerate(answer['options']):
                print(f"{space}{chr(65 + idx)}: {option}")
            if show_answers:
                correct_option_idx = answer['options'].index(answer['correct'])
                print(f"{space}Correct Answer: {chr(65 + correct_option_idx)}: {answer['correct']}\n")
        else:  # sentence format
            if show_answers:
                print(f"{space}Answer: {answer}\n")


def save_qa_to_txt(qa_list: List[Mapping[str, str]], file_path: str) -> None:
    """Saves a list of generated questions and answers to a text file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for i, qa in enumerate(qa_list):
            space = " " * (3 if i < 9 else 4)
            file.write(f"{i + 1}) Question: {qa['question']}\n")
            answer = qa["answer"]
            if isinstance(answer, dict) and 'options' in answer and 'correct' in answer:
                for idx, option in enumerate(answer['options']):
                    file.write(f"{space}{idx + 1}. {option}\n")
                correct_option_idx = answer['options'].index(answer['correct']) + 1
                file.write(f"{space}Correct Answer: {correct_option_idx}. {answer['correct']}\n")
            else:
                file.write(f"{space}Answer: {answer}\n")
            file.write("\n")
