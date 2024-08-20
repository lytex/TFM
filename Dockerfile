FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
RUN ln -s /usr/share/zoneinfo/UTC /etc/localtime
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tree software-properties-common wget graphviz htop && \
    add-apt-repository ppa:deadsnakes/ppa && add-apt-repository ppa:flexiondotorg/nvtop
RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y python3.7 nvtop
COPY --from=python:3.7-slim /usr/local/lib/python3.7/distutils /usr/lib/python3.7/distutils
RUN wget https://bootstrap.pypa.io/pip/3.7/get-pip.py 
RUN python3.7 get-pip.py --target /usr/local/lib/python3.7/dist-packages
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install -r requirements.txt --target /usr/local/lib/python3.7/dist-packages
RUN useradd --create-home -u 42420 ovh

# USER ovh
RUN mkdir -p /home/ovh/code
WORKDIR /home/ovh/code
COPY *.csv /home/ovh/code
COPY *.py /home/ovh/code
COPY entrypoint.sh /home/ovh/code

ENTRYPOINT ["/bin/bash", "/home/ovh/code/entrypoint.sh"]

