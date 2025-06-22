FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY iris_model iris_model

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
