
FROM python:3.7.17-slim-bullseye
RUN sed -i "s/bullseye main/bullseye main contrib non-free/" /etc/apt/sources.list
RUN apt-get update && apt-get install -y nvidia-cuda-toolkit nvidia-cuda-dev 

RUN mkdir -p /code
WORKDIR /code

COPY requirements.txt /code
RUN pip install -r requirements.txt
COPY *.csv /code
COPY *.py /code

ENTRYPOINT ["python", "optuna_trial.py"]

