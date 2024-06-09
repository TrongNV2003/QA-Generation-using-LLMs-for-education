import json

def extract_questions_answers(json_file, output_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in [item for item in data if item["question_type"] == "multiple_choice"]:
            context = item['context']
            question = item['question']
            options = item['options']
            answer_index = ord(item['answer']) - ord('A')  # Convert letter to index
            answer = options[answer_index]
            
            # answer = item['answer']
            f.write(f"{question} || {answer} \n")

# Đường dẫn đến tập dữ liệu JSON và tệp văn bản đầu ra
json_file = 'datasets/test/qg_test.json'
output_file = 'output_questions_answers.txt'

# Gọi hàm để trích xuất câu hỏi và câu trả lời
extract_questions_answers(json_file, output_file)
