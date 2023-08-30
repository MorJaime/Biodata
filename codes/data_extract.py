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
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
#from keras.utils import np_utils
#from keras.utils.vis_utils import model_to_dot
from keras.callbacks import EarlyStopping
from IPython.display import SVG

#import tensorflow_probability as tfp

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, normalize
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

from ipywidgets import FloatProgress
from IPython.display import display

#Label paths (Relative)

OMIZU_PATH = os.environ["OMIZU_PATH"]
UMINEKO_PATH = os.environ["UMINEKO_PATH"]

def find_csv_filenames(path_to_dir, suffix=".csv", year = '2022'):
    filenames = os.listdir(path_to_dir)
    filepaths = []
    for filename in filenames:
        if filename.endswith( suffix ):
            if filename.__contains__(year):
                filepaths.append(os.path.join(path_to_dir,filename))
    return filepaths

def separate_by_sensor(filename, save_folder, sensor='acc', time_format="%Y%m%d_%H:%M:%S.%f"):
    data = pd.read_csv(filename, parse_dates=["timestamp"])    
    data["timestamp"] = pd.to_datetime(data["timestamp"],format=time_format)
    
    if sensor == 'acc':
        new_df = data.drop(['logger_id', 'latitude', 'longitude', 'gps_status', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'illumination', 'pressure', 'temperature'],axis=1)
    else:
        new_df = data
        
    name = filename[-11:-4]
    name = name.replace('_','')
    name = name.replace('00','')
    save_name = os.path.join(save_folder,name+'_acc.csv')
    new_df.to_csv(save_name,index=False)
    
    data_df = pd.read_csv(save_name)
    l = []
    for i in range(len(data)):
        l.append(data_df['timestamp'][i].replace('+00:00',''))
    data_df['timestamp'] = l
    #data_df['timestamp'][0] = data_df['timestamp'][0]+'.000000'
    
    #data_df = pd.to_datetime(data_df["timestamp"],format=time_format)
    
    #data_df = time_change(data_df)
    
    #data_df.to_csv(os.path.join(save_folder,name+'t_acc.csv'),index=False)

def main(args):
    bird = args['bird']
    outdir = args['out_dir']
    year = args['year']
    raw_path_o = os.path.join(OMIZU_PATH,'raw')
    raw_path_u = os.path.join(UMINEKO_PATH,'raw')

    if bird == 'omizunagidori' or bird == 'O':
        save_folder = os.path.join(outdir, 'omizunagidori', year)
        raw_paths = find_csv_filenames(raw_path_o, suffix = ".csv", year = year)
    else:
        save_folder = os.path.join(outdir, 'umineko', year)
        raw_paths = find_csv_filenames(raw_path_u, suffix=".csv", year = year)
    
    for path in raw_paths:
        separate_by_sensor(path,save_folder)
        print('path: ', path)
        print('save folder: ', save_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bird', help="Bird kind to extract either omizunagidoriu/umineko (O/U)", required=True)
    parser.add_argument('-o', '--out-dir', help="Path to the output directory", required=True)
    parser.add_argument('-y', '--year', help="Year to extract", required=True)

    args = vars(parser.parse_args())
    main(args)