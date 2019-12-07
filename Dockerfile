FROM tensorflow/tensorflow:1.15.0-py3
MAINTAINER seongahjo "seongside@gmail.com"
RUN apt-get install -y libsndfile1
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 80
ENTRYPOINT ["python"]
CMD ["web.py"]