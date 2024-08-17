FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
RUN ln -s /usr/share/zoneinfo/UTC /etc/localtime
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common wget graphviz && \
    add-apt-repository ppa:deadsnakes/ppa
RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y python3.7
COPY --from=python:3.7-slim /usr/local/lib/python3.7/distutils /usr/lib/python3.7/distutils
RUN wget https://bootstrap.pypa.io/pip/3.7/get-pip.py 
RUN python3.7 get-pip.py --target /usr/local/lib/python3.7/dist-packages
RUN useradd --create-home appuser
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install -r requirements.txt --target /usr/local/lib/python3.7/dist-packages

USER appuser
RUN mkdir -p /home/appuser/code
WORKDIR /home/appuser/code
COPY *.csv /home/appuser/code
COPY *.py /home/appuser/code

ENTRYPOINT ["python3.7", "optuna_trial.py"]

