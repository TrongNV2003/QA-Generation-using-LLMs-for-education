from questiongenerator import QuestionAnswerGenerator
from questiongenerator import print_qa
from questiongenerator import save_qa_to_txt

qg = QuestionAnswerGenerator()

with open('articles/philosophy.txt', 'r',encoding='utf-8') as a:
    article = a.read()

qa_list = qg.generate(
    article,
    num_questions=3,
    answer_style='multiple_choice'
)

print_qa(qa_list, show_answers=True)


# qg.save_questions_to_file(qa_list, "questions.txt")

# Sử dụng hàm để lưu output
# output_file_path = "generated_questions.txt"
# save_qa_to_txt(qa_list, output_file_path)
# print(f"Output đã được lưu vào file '{output_file_path}'")