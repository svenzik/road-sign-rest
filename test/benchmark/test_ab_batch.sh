#!/bin/bash

BATCH_SIZE=100
#BATCH_SIZE=50
RESULT_FILE=test_ab_result.csv

#Swarm
#URL=http://192.168.83.200:9001/rest2fs
#zero
#URL=http://raspberrypi-201:9002/rest2fs
#PI3
#URL=http://raspberrypi-m2:9002/rest2fs

### REAL TIME
#URL=http://192.168.83.200:9001/s2img
#URL=http://192.168.83.200:9001/s2speed
URL=http://192.168.83.200:9002/s2speed

echo 'file_name	starttime	seconds	ctime	dtime	ttime	wait' > $RESULT_FILE


for json_file in "$@"
do
	echo testing file: $json_file
	
	CSV_FILE=${json_file}_1t.csv
	ab -c 1 -g ${CSV_FILE} -n $BATCH_SIZE -T "application/json" -p ${json_file} ${URL}
	FILE_NAME=`basename $CSV_FILE`
	tail -n +2 $CSV_FILE | sed -e "s/^/$FILE_NAME\t/g" >> $RESULT_FILE
	
	CSV_FILE=${json_file}-2t.csv
	###ab -c 2 -g ${CSV_FILE} -n $BATCH_SIZE -T "application/json" -p ${json_file} ${URL}
	FILE_NAME=`basename $CSV_FILE`
	###tail -n +2 $CSV_FILE | sed -e "s/^/$FILE_NAME\t/g" >> $RESULT_FILE
done

