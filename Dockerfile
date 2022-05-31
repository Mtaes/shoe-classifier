FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
WORKDIR /shoes/src
RUN apt-get update && apt-get install -y python3 python3-pip
COPY ./requirements.txt ../
RUN pip3 install --user -r ../requirements.txt
COPY ./src/prepare_data.py ./
RUN python3 prepare_data.py
COPY ./data/Shoes ../data/Shoes
COPY src/* .
