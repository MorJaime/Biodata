FROM jupyter/tensorflow-notebook

USER root
RUN apt update && apt-get install -y \
      nfs-common \
      cifs-utils \
      rpcbind \
      libopencv-dev \
      python3-opencv \
      && pip3 install opencv-python opencv-contrib-python

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
VOLUME  ${HOME}/storage/export


#ENV MODEL_DIR=/home/jovyan/my-model
#ENV MODEL_FILE_LDA=clf_lda.joblib
#ENV MODEL_FILE_NN=clf_nn.joblib
#ENV OUTPUT_PATH=/home/jovyan/my-data

RUN pip install joblib

#COPY train.csv ./train.csv
#COPY test.csv ./test.csv

#COPY train.py ./train.py
#COPY inference.py ./inference.py