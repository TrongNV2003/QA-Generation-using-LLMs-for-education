from questiongenerator import QuestionAnswerGenerator
from questiongenerator import print_qa
from questiongenerator import save_qa_to_txt

qg = QuestionAnswerGenerator()

with open('articles/philosophy.txt', 'r',encoding='utf-8') as a:
    article = a.read()

qa_list = qg.generate(
    article,
    num_questions=5,
    answer_style='multiple_choice'
)

print_qa(qa_list, show_answers=True)


# Sử dụng hàm để lưu output
# output_file_path = "generated_questions.txt"
# save_qa_to_txt(qa_list, output_file_path)
# print(f"Output đã được lưu vào file '{output_file_path}'")

# from questiongenerator import QuestionAnswerGenerator
# from questiongenerator import print_qa
# import json

# qg = QuestionAnswerGenerator()

# # Đọc dữ liệu từ tập test.json
# with open('datasets/test/qg_test.json', 'r', encoding='utf-8') as f:
#     test_data = json.load(f)

# all_qa_list = []

# # Sinh câu hỏi và câu trả lời cho từng context có question_type là 'multiple_choice'
# for item in test_data:
#     if item['question_type'] == 'multiple_choice':
#         context = item['context']
#         qa_list = qg.generate(
#             context,
#             num_questions=1,
#             answer_style='multiple_choice'
#         )
#         all_qa_list.extend(qa_list)

# # Hàm lưu các cặp câu hỏi-câu trả lời vào tệp dưới dạng question; answer
# def save_qa_to_txt(qa_list, file_path):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for qa in qa_list:
#             question = qa['question']
#             correct_answer = qa['answer']['correct']
#             f.write(f"{question}; {correct_answer}\n")

# # Sử dụng hàm để lưu output
# output_file_path = "generated_questions.txt"
# save_qa_to_txt(all_qa_list, output_file_path)