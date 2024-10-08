# from flask import Flask, render_template, request, jsonify
# from questiongenerator import QuestionAnswerGenerator
# import threading
# import uuid

# qg = QuestionAnswerGenerator()
# app = Flask(__name__)

# tasks = {}

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate_questions', methods=['POST'])
# def generate_questions():
#     data = request.json
#     article = data['article']
#     num_questions = int(data['num_questions'])
#     answer_style = data['answer_style']

#     task_id = str(uuid.uuid4())
#     tasks[task_id] = {
#         'status': 'PENDING',
#         'result': None
#     }

#     thread = threading.Thread(target=generate_questions_task, args=(task_id, article, num_questions, answer_style))
#     thread.start()

#     return jsonify({"task_id": task_id}), 202

# @app.route('/task_status/<task_id>')
# def task_status(task_id):
#     task = tasks.get(task_id, None)
#     if not task:
#         return jsonify({'state': 'FAILURE', 'status': 'Task not found'})

#     return jsonify(task)

# def generate_questions_task(task_id, article, num_questions, answer_style):
#     # Chuyển mô hình sang GPU (nếu chưa)
#     qg.qg_model.to(qg.device)
#     qg.mcq_qg_model.to(qg.device)
#     qg.qd_model.to(qg.device)
    
#     tasks[task_id]['status'] = 'IN_PROGRESS'
#     qa_list = qg.generate(article, num_questions=num_questions, answer_style=answer_style)
#     tasks[task_id]['status'] = 'SUCCESS'
#     tasks[task_id]['result'] = qa_list

# if __name__ == '__main__':
#     app.run(debug=True)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from questiongenerator import QuestionAnswerGenerator
import threading
import uuid

qg = QuestionAnswerGenerator()
app = FastAPI()

templates = Jinja2Templates(directory="templates")  # Thư mục chứa template HTML

tasks = {}

# Model để nhận dữ liệu từ request
class GenerateQuestionsRequest(BaseModel):
    article: str
    num_questions: int
    answer_style: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate_questions", status_code=202)
async def generate_questions(request_data: GenerateQuestionsRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'PENDING',
        'result': None
    }

    thread = threading.Thread(target=generate_questions_task, args=(task_id, request_data.article, request_data.num_questions, request_data.answer_style))
    thread.start()

    return {"task_id": task_id}

@app.get("/task_status/{task_id}")
async def task_status(task_id: str):
    task = tasks.get(task_id, None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task

def generate_questions_task(task_id, article, num_questions, answer_style):
    # Chuyển mô hình sang GPU (nếu chưa)
    qg.qg_model.to(qg.device)
    qg.mcq_qg_model.to(qg.device)
    qg.qd_model.to(qg.device)

    tasks[task_id]['status'] = 'IN_PROGRESS'
    qa_list = qg.generate(article, num_questions=num_questions, answer_style=answer_style)
    tasks[task_id]['status'] = 'SUCCESS'
    tasks[task_id]['result'] = qa_list

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
