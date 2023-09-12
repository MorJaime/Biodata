import os
import glob
import math
import pandas as pd
import numpy as np
import itertools
import random
import requests
import xml.etree.ElementTree as ET
import csv
import argparse
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import SVG

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, normalize
from sklearn.preprocessing import MinMaxScaler
from logging import getLogger

warnings.filterwarnings('ignore')
logger = getLogger(__name__)

def time_change(acc_df,column='timestamp'):
    df=acc_df[column]
    df1=df.astype(np.int64)
    acc_df[column]=df1/1000000
    acc_df[column] = acc_df[column].map(lambda x: int(x))
    return acc_df

def setup_dir(path: str):
    """ Setup write directory

    Args.
    ------
    - path: path to write folder

    """
    if not os.path.isdir(path):
        # If selected DIR does not exist, create it.
        os.makedirs(path)
        if os.path.isdir(path):
            logger.info(f"Created Dir: {path}")

    return

def find_xml_filenames(path_to_dir, suffix=".xml"):
    filenames = os.listdir(path_to_dir)
    filepaths = []
    for filename in filenames:
        if filename.endswith( suffix ):
            filepaths.append(os.path.join(path_to_dir,filename))
    return filepaths

def find_csv_filenames(path_to_dir, suffix=".csv", year = '2022'):
    filenames = os.listdir(path_to_dir)
    filepaths = []
    for filename in filenames:
        if filename.endswith( suffix ):
            if filename.__contains__(year):
                filepaths.append(os.path.join(path_to_dir,filename))
    return filepaths

def get_activity_index(act_dict):
    labels = []
    indices = []
    for k,v in act_dict.items():
        indices.append(k)
        labels.append(v)
    return labels, indices

def num_labels(acc_df,lab):
    
    new_df=acc_df
    new_df['l_val']=100

    i=0
    for a in lab.label:
        #print(a,':')
        new_df.loc[new_df['label'] == a, 'l_val'] = i
        #print(new_df)
        i=i+1
    
    new_df.fillna(100,inplace=True)   
    
    return new_df