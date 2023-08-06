FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt update && apt install -y \
	curl \
	tar \
	git \
	wget \
	vim \
	unzip

RUN conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath

RUN conda install -c bottler nvidiacub

RUN mkdir -p /tmp/ && cd /tmp/ \
	&& curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz \
	&& tar xzf 1.10.0.tar.gz \
	&& export CUB_HOME=$PWD/cub-1.10.0 \
	&& conda install pytorch3d -c pytorch3d \
	&& rm 1.10.0.tar.gz

RUN pip install -U pip setuptools

RUN pip install \
	numpy \
	open3d \
	einops \
	hesiod \
	fvcore \
	tensorboard \
	trimesh \
	cython \
	networkx \
	chumpy

RUN cd /tmp/ \
	&& git clone https://github.com/liren2515/DrapeNet.git \
	&& cd DrapeNet/meshudf/ \
	&& python setup.py build_ext --inplace \
	&& mv /tmp/DrapeNet/ /DrapeNet/

WORKDIR /DrapeNet
