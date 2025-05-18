FROM python:3.11-slim

WORKDIR /app
COPY ./app /app

RUN pip install --no-cache-dir numpy matplotlib

CMD ["python", "main.py"]