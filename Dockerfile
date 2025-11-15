FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["uvicorn", "housing.app:app", "--host", "0.0.0.0", "--port", "8000"]