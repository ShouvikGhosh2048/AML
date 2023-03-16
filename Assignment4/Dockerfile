# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /python-docker

RUN pip3 install scikit-learn Flask

COPY app.py app.py
COPY score.py score.py
COPY train.csv train.csv

CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0"]