
version=latest
if [ ! -z "$1" ]; then
  version = $1
fi

git pull

docker build \
  -t road-sign-rest \
  -t svenzik/road-sign-rest \
  -t svenzik/road-sign-rest:$version \
  .

docker service update --image svenzik/road-sign-rest:$version py-rsr
