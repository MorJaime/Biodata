#FROM continuumio/anaconda3
FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

USER root
RUN apt-get install -y gnupg2
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update && apt-get install -y \
      nfs-common \
      cifs-utils \
      rpcbind \
      && pip install --upgrade pip
#      libopencv-dev \
#     python3-opencv \
#      && pip3 install opencv-python opencv-contrib-python

# -- Add User (ID: bob, PW: bob) --
ENV USER bob
ENV HOME /home/${USER}
ENV SHELL /bin/bash
RUN useradd -m ${USER} \
    && gpasswd -a ${USER} sudo \
    && echo "bob:P2ssword" | chpasswd
USER ${USER}
WORKDIR /home/bob
ENV HARDS_LOG_LEVEL DEBUG
ENV PATH ${PATH}:/home/bob/.local/bin

# -- Setup Directories --
USER bob
WORKDIR ${HOME}
VOLUME  ${HOME}/biodata
VOLUME  ${HOME}/storage

#RUN conda init bash
#RUN conda create -n biodata-env python=3.10
#SHELL ["conda", "run", "-n", "biodata-env", "/bin/bash", "-c"]
#RUN conda install -c conda-forge cudatoolkit=11.8.0
#RUN python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
#RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
#RUN echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#RUN echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#RUN source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#RUN python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#RUN conda install tensorflow=2.10.*=gpu_*
RUN pip install joblib

ENV BIODATA_PATH ${HOME}/biodata
ENV OMIZU_PATH ${HOME}/storage/export/Omizunagidori
ENV UMINEKO_PATH ${HOME}/storage/export/Umineko
ENV CSVWRITE_PATH ${HOME}/storage/database
ENV LABELS_PATH ${HOME}/storage/database/labels
ENV O_WRITE_PATH ${HOME}/storage/database/omizunagidori
ENV U_WRITE_PATH ${HOME}/storage/database/umineko

#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]