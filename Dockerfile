FROM jupyter/tensorflow-notebook

RUN mkdir my-model
VOLUME /home/jovyan/my-model
VOLUME /home/jovyan/my-data

ENV MODEL_DIR=/home/jovyan/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib
ENV OUTPUT_PATH=/home/jovyan/my-data

RUN pip install joblib

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py