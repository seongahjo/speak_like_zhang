FROM tensorflow/tensorflow:1.15.0
MAINTAINER seongahjo "seongside@gmail.com"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt