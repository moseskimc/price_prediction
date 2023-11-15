FROM ubuntu:latest
FROM python:3.11-slim-buster

RUN apt update
RUN apt install python3 -y
RUN apt install -y libgl1-mesa-glx
RUN apt-get update && apt-get -y install libglib2.0-0; apt-get clean

WORKDIR /usr/app/src

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "src/train.py"]