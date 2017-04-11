#FROM hypriot/rpi-python
#FROM resin/rpi-raspbian:latest
FROM rickryan/rpi-jessie-opencv3.2:latest

#RUN apt-get update -y
#RUN apt-get install -y \
#	python-pip 
#	python-dev \
#	build-essential 
#	python-opencv \
#	python-numpy

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]

EXPOSE 9001
CMD ["app.py"]

