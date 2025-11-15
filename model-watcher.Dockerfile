FROM python:3.9-slim

WORKDIR /app

COPY model_watcher.py .

RUN pip install kubernetes

CMD ["python", "model_watcher.py"]