# syntax=docker/dockerfile:1

FROM python:3.8
ENV PYTHONUNBUFFERRED=1
WORKDIR /ai4pbs
COPY ./requirements.txt requirements.txt 
RUN pip install -r requirements.txt