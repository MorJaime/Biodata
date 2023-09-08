FROM continuumio/anaconda3
#FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

USER root
RUN apt update && apt-get install -y \
      nfs-common \
      cifs-utils \
      rpcbind
#      libopencv-dev \
#      python3-opencv \
#      && pip3 install opencv-python opencv-contrib-python

#
EXPOSE 8888 6006

#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

COPY environment.yml .
COPY requirements.txt .
RUN conda env create -f environment.yml
#RUN conda create --name biodata-env --file requirements.txt
ENV CONDA_ENV biodata-env

SHELL ["conda", "run", "-n", "biodata-env", "/bin/bash", "-c"]
#


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

RUN pip install joblib

ENV BIODATA_PATH ${HOME}/biodata
ENV OMIZU_PATH ${HOME}/storage/export/Omizunagidori
ENV UMINEKO_PATH ${HOME}/storage/export/Umineko
ENV CSVWRITE_PATH ${HOME}/storage/database
ENV LABELS_PATH ${HOME}/storage/database/labels
ENV O_WRITE_PATH ${HOME}/storage/database/omizunagidori
ENV U_WRITE_PATH ${HOME}/storage/database/umineko