nvidia-docker run -it \
	-p 8888:8888 -p 8889:8889 -p 8890:8890 \
	-v $HOME/Hive/MOOCs/dlnd:/root/sharedfolder \
	gcr.io/tensorflow/tensorflow:1.0.0-gpu-py3 \
	bash

cd /root/sharedfolder
