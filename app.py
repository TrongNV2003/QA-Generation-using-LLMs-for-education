from flask import Flask, render_template, request
from questiongenerator import QuestionGenerator

qg = QuestionGenerator()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    article = request.form['article']
    num_questions = int(request.form['num_questions'])
    answer_style = request.form['answer_style']

    qa_list = qg.generate(article, num_questions=num_questions, answer_style=answer_style)
    
    if answer_style == 'sentences':
        return render_template('questions.html', qa_list=qa_list)
    elif answer_style == 'multiple_choice':
        return render_template('questions_mcq.html', qa_list=qa_list)
    else:
        return "Invalid answer style"

if __name__ == '__main__':
    app.run(debug=True)
