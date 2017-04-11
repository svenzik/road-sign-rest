#FROM hypriot/rpi-python
#FROM resin/rpi-raspbian:latest
FROM rickryan/rpi-jessie-opencv3.2:latest

#RUN apt-get update -y && apt-get install -y \
#	python-flask \
#	python-click
#	python-pip 
#	python-dev \
#	build-essential 
#	python-opencv \
#	python-numpy

COPY . /app
WORKDIR /app
#RUN pip search opencv
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]

EXPOSE 9001
CMD ["app.py"]

