ARG BASE_IMAGE_TYPE=cpu
FROM python:3.8

RUN apt-get update
RUN apt-get install wget ffmpeg libsm6 libxext6  -y

ENV DATA_DIR=/app/FedKNOW/data
RUN mkdir -p $DATA_DIR
ADD data DATA_DIR

WORKDIR /app
# update pip
RUN pip3 install --upgrade pip

ADD FedKNOW/docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# Scripts needed for Flower client
ADD flwr /app/flwr
ADD FedKNOW/multi /app/FedKNOW/multi
ADD FedKNOW/utils /app/FedKNOW/utils
ADD FedKNOW/scripts /app/FedKNOW/scripts
ADD FedKNOW/data /app/FedKNOW/data
ADD FedKNOW/dataset /app/FedKNOW/dataset
ADD FedKNOW/models /app/FedKNOW/models
ADD FedKNOW/test.py /app/FedKNOW

ENTRYPOINT ["python3","-m","FedKNOW.multi.main_WEIT"]
