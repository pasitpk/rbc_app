# syntax=docker/dockerfile:1

FROM python:3.8
ENV PYTHONUNBUFFERRED=1
WORKDIR /app
COPY ./requirements.txt requirements.txt 
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt