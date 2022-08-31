#FROM nvcr.io/nvidia/pytorch:20.08-py3
FROM pytorch/pytorch
# Install some basic utilities and python
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse' > /etc/apt/sources.list && \
echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list && \
echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse' >> /etc/apt/sources.list && \
echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list
RUN apt-get update 
RUN apt-get install -y python3-pip python3-dev 
RUN cd /usr/local/bin
RUN ln -s /usr/bin/python3 python
RUN pip3 install --upgrade pip

# install nnunet
# if you used other framework, mask sure to install the used package and add related inference code
COPY ./   ./
RUN pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
# for single model inference
# ADD parameters /parameters/
# ADD run_inference.py ./


RUN groupadd -r myuser -g 433 && \
    useradd -u 431 -r -g myuser -s /sbin/nologin -c "Docker image user" myuser

#RUN mkdir /input_nifti && mkdir /output_nifti && chown -R myuser /input_nifti && chown -R myuser /output_nifti

USER myuser

#CMD python3 ./run_inference.py
# or CMD python3 ./run_inference_ensembling.py
