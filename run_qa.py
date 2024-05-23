from questiongenerator import QuestionAnswerGenerator
from questiongenerator import DistractorGenerator
from questiongenerator import print_qa
from questiongenerator import save_qa_to_txt
qa_generator = QuestionAnswerGenerator()
distractor_generator = DistractorGenerator()

with open('articles/philosophy2.txt', 'r',encoding='utf-8') as a:
    context = a.read()

# Generate questions and answers
qa_pairs = qa_generator.generate(context, answer_style="multiple_choice")

# Generate distractors for each QA pair
for qa in qa_pairs:
    question = qa["question"]
    correct_answer = qa["answer"]
    distractors = distractor_generator.generate_distractors(correct_answer, context)
    qa["answer"] = {
        "options": distractors + [correct_answer],
        "correct": correct_answer
    }
print_qa(qa_pairs)



# print(qa_list)
# qg.save_questions_to_file(qa_list, "questions.txt")

# # Sử dụng hàm để lưu output
# output_file_path = "generated_questions.txt"
# save_qa_to_txt(qa_list, output_file_path)
# print(f"Result has been saved in file: {output_file_path}")