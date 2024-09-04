# 1. Chọn base image
FROM python:3.11-slim

# 2. Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# 3. Copy file requirements.txt trước để cài đặt các thư viện trước khi copy toàn bộ mã nguồn
COPY requirements.txt .

# 4. Cập nhật pip và cài đặt các thư viện cần thiết từ file requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy tất cả file từ thư mục hiện tại (host) vào thư mục /app trong container
COPY . .

# 6. Định nghĩa command mặc định để chạy ứng dụng
CMD ["python", "run_qa.py"]
