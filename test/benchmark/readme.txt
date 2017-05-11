#base
#Swarm
#URL=http://raspberrypi-m2:9001/rest2fs
#zero
#URL=http://raspberrypi-201:9002/rest2fs
#PI3
#URL=http://raspberrypi-m2:9002/rest2fs

#s2img
#URL=http://raspberrypi-m2:9001/s2img
#zero
#URL=http://raspberrypi-201:9002/s2img
#PI3
#URL=http://raspberrypi-m2:9002/s2img

#s2speed
#URL=http://raspberrypi-m2:9001/s2speed
#zero
#URL=http://raspberrypi-201:9002/s2speed
#PI3
#URL=http://raspberrypi-m2:9002/s2speed


#PI3 cluster
test cluster and 1PI3 
	base method
	120
	france
120 & france separate#
../benchmark/test_ab_methods.sh http://raspberrypi-m2:9001/rest2fs http://raspberrypi-m2:9002/rest2fs http://raspberrypi-m2:9001/s2img http://raspberrypi-m2:9002/s2img http://raspberrypi-m2:9001/s2speed http://raspberrypi-m2:9002/s2speed

#Zero cluster
test cluster and 1PI3 
	base method
	120
	france
../benchmark/test_ab_methods.sh http://raspberrypi-m2:9001/rest2fs http://raspberrypi-201:9002/rest2fs http://raspberrypi-m2:9001/s2img http://raspberrypi-201:9002/s2img http://raspberrypi-m2:9001/s2speed http://raspberrypi-201:9002/s2speed

