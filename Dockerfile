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

RUN pip install joblib

ENV OMIZU_PATH ${HOME}/storage/export/Omizunagidori
ENV UMINEKO_PATH ${HOME}/storage/export/Umineko
ENV BIODATA_PATH ${HOME}/biodata
ENV LABELS_PATH ${HOME}/biodata/database/labels
ENV CSVWRITE_PATH ${HOME}/biodata/database/csv

