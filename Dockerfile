FROM tensorflow/tensorflow:1.15.0-py3
MAINTAINER seongahjo "seongside@gmail.com"
RUN apt-get install -y libsndfile1
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python"]
CMD ["web.py"]