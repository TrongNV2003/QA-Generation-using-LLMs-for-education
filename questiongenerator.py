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
        self.qg_tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")
        self.qg_tokenizer.add_special_tokens({"sep_token": "<sep>"})
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        QG_PRETRAINED = "t5-base-question-generator"
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()
        
        MCQ_QG_PRETRAINED = "t5-base-question-mcq-generator"
        self.mcq_qg_model = AutoModelForSeq2SeqLM.from_pretrained(MCQ_QG_PRETRAINED)
        self.mcq_qg_model.to(self.device)
        self.mcq_qg_model.eval()
        
        QD_PRETRAINED = "t5-base-distractor-generator"
        self.qd_model = AutoModelForSeq2SeqLM.from_pretrained(QD_PRETRAINED)
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

        # Chọn top N câu hỏi
        if len(questions) > num_questions:
            questions, answers = questions[:num_questions], answers[:num_questions]

        return inputs, questions, answers
    
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
            qg_input = f"Essay: {sentences}"
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
                correct_answer = correct_answer.replace("Answer:", "").replace("Question: ", "").strip()
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
            qg_input = f"Multiple choice: {sentences}"
            encoded_input = self._encode_qg_input(qg_input)
            outputs = self.mcq_qg_model.generate(
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
                correct_answer = correct_answer.replace("Answer: ", "").replace("Question: ", "").strip()
                question_answer_split = correct_answer.split(self.qg_tokenizer.sep_token)
                if len(question_answer_split) == 2:
                    # valid Question + Answer output
                    question, answer = question_answer_split[0].strip(), question_answer_split[1].strip()
                    inputs.append(qg_input)
                    questions.append(question)
                    answers.append(answer)

        return inputs, questions, answers
    
    @torch.no_grad()
    def _generate_distractors(self, question: str, correct_answer: str, context: str) -> List[str]:
        distractors = set()
        attempts = 0
        distractor_input = f"Question: {question} {self.qg_tokenizer.sep_token} Answer: {correct_answer} {self.qg_tokenizer.sep_token} Multiple choice: {context}"

        while len(distractors) < 3 and attempts < 10:
            attempts += 1
            input_ids = self._encode_qg_input(distractor_input)
            outputs = self.qd_model.generate(
                input_ids=input_ids["input_ids"], 
                max_new_tokens=128, 
                temperature=0.7,
                num_beams=5,
                num_return_sequences=3,
                do_sample=True
                )
            for output in outputs:
                generated_text = self.qg_tokenizer.decode(output, skip_special_tokens=True)
                generated_text = generated_text.split(self.qg_tokenizer.sep_token)
                for distractor in generated_text:
                    if distractor.lower() != correct_answer.lower() and distractor not in distractors:
                        distractors.add(distractor)
                    if len(distractors) >= 3:
                        break

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

                # Ensure distractors are unique
                distractors = list(set(distractors))
                options = [answer] + distractors[:3]  # ensure we only take the first 3 unique distractors
                random.shuffle(options)
                qa_list.append({
                    "question": question.split("?")[0] + "?",
                    "answer": {
                        "options": options,
                        "correct": answer,
                        "context": context
                    }
                })
        else:
            qa_list = [{"question": question.split("?")[0] + "?", "answer": answer} for question, answer in zip(questions, answers)]
        return qa_list


def print_qa(qa_list: List[Mapping[str, str]], show_answers: bool = True) -> None:
    """Formats and prints a list of generated questions and answers."""
    for i, qa in enumerate(qa_list):
        space = " " * (3 if i < 9 else 4)
        print(f"{i + 1}) Q: {qa['question']}")
        answer = qa["answer"]
        if isinstance(answer, dict):  # multiple choice format
            for idx, option in enumerate(answer['options']):
                print(f"{space}{chr(65 + idx)}: {option}")
            if show_answers:
                correct_option_idx = answer['options'].index(answer['correct'])
                print(f"{space}Correct Answer: {chr(65 + correct_option_idx)}: {answer['correct']}\n")
        else:  # sentence format
            if show_answers:
                print(f"{space}A: {answer}\n")


def save_qa_to_txt(qa_list: List[Mapping[str, str]], file_path: str) -> None:
    """Saves a list of generated questions and answers to a text file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for i, qa in enumerate(qa_list):
            space = " " * (3 if i < 9 else 4)
            file.write(f"{i + 1}) Q: {qa['question']}\n")
            answer = qa["answer"]
            if isinstance(answer, dict) and 'options' in answer and 'correct' in answer:
                file.write(f"{space}Context: {answer['context']}\n")
                for idx, option in enumerate(answer['options']):
                    file.write(f"{space}{idx + 1}. {option}\n")
                correct_option_idx = answer['options'].index(answer['correct']) + 1
                file.write(f"{space}Correct Answer: {correct_option_idx}. {answer['correct']}\n")
            else:
                file.write(f"{space}A: {answer}\n")
            file.write("\n")
