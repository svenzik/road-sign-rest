FROM hypriot/rpi-python


ADD ./package.json ./
RUN apt-get update && apt-get install -y
ADD ./app.js ./
EXPOSE 9000

CMD ["node", "./app.js"]

