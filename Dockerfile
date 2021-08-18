# syntax=docker/dockerfile:1

FROM python:3.8
ENV PYTHONUNBUFFERRED=1
WORKDIR /app
COPY ./requirements.txt requirements.txt 
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python