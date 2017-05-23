
HOST=http://localhost:9001
METHOD=$1
IMG_FILE=$2
FILE=`./create-test-pacakge-from-image $IMG_FILE`

URL=${HOST}/${METHOD}
DATE=`date +%Y%m%d_%H%M%S`
JSON_FILE=${DATE}.json
IMG_FILE=${DATE}.jpg

echo Connecting to $URL
curl ${URL} -H "Content-Type: application/json" -d @${FILE} > $JSON_FILE
grep 'image_path' $JSON_FILE | sed -e 's/image_path/speed/'
grep '\"image\"' $JSON_FILE | sed -e 's/.*: *//;s/[",]//g' | base64 -d > $IMG_FILE
