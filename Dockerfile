
FROM python:3.7.17-slim-bullseye

RUN mkdir -p /code
WORKDIR /code

COPY requirements.txt /code
RUN pip install -r requirements.txt
COPY *.csv /code
COPY *.py /code


