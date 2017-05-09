
version=latest
if [ ! -z "$1" ]; then
  version=$1
fi

git pull

docker build \
  -t road-sign-rest \
  -t svenzik/road-sign-rest \
  -t svenzik/road-sign-rest:$version \
  .

docker push svenzik/road-sign-rest:$version
docker push svenzik/road-sign-rest

echo Running: docker service update --image svenzik/road-sign-rest:$version py-rsr
docker service update --image svenzik/road-sign-rest:$version py-rsr

docker service ps py-rsr
