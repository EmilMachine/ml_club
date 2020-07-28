FROM python:3.8-slim AS train

COPY ./data/titanic/* ./data/titanic/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app/train.py /app/train.py
RUN python3 /app/train.py