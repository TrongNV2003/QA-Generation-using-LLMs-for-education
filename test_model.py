from questiongenerator import QuestionAnswerGenerator
from questiongenerator import print_qa
from questiongenerator import save_qa_to_txt
qg = QuestionAnswerGenerator()

with open('articles/philosophy2.txt', 'r',encoding='utf-8') as a:
    article = a.read()

qa_list = qg.generate(
    article,
    num_questions=3,
    answer_style='sentences'
)
# print_qa(qa_list, show_answers=False)
print_qa(qa_list, show_answers=True)
# print(qa_list)
# qg.save_questions_to_file(qa_list, "questions.txt")

# # Sử dụng hàm để lưu output
# output_file_path = "generated_questions.txt"
# save_qa_to_txt(qa_list, output_file_path)
# print(f"Result has been saved in file: {output_file_path}")