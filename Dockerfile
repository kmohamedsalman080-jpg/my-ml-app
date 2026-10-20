# 1. Base Image
FROM python:3.10-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Requirements & Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Files & Model
COPY main.py .
COPY server_model.pkl .
COPY .env .

# 5. Expose Application Port
EXPOSE 8000

# 6. Run FastAPI App
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]