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
CSVWRITE_PATH = os.environ["CSVWRITE_PATH"]

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

def join_by_year(read_dir):
    pre_df_fn = os.listdir(read_dir)[0]
    pre_df = pd.read_csv(os.path.join(read_dir, pre_df_fn),parse_dates=["timestamp"])
    all_bird_df = pd.DataFrame(columns = pre_df.columns)
    all_bird_lst = []
    for file in os.listdir(read_dir):
        if file.endswith(".csv"):
            path = os.path.join(read_dir, file)
            print(os.path.join(read_dir, file))
            df = pd.read_csv(path,parse_dates=["timestamp"])
            #all_bird_lst.append(df)
            all_bird_df = all_bird_df._append(df, ignore_index=True)
            #all_bird_df = pd.concat([all_bird_df,df])

    #all_bird_df = pd.DataFrame(all_bird_lst, columns = pre_df.columns)
    #all_bird_df = pd.concat(all_bird_lst)

    birds = list(all_bird_df.drop_duplicates(subset=['animal_tag'],keep = 'first')['animal_tag'])
    print(birds)
    
    all_bird_df.to_csv(os.path.join(read_dir, 'all_bird_df.csv'),index=False)

    print(all_bird_df)

    all_bird_df.drop(['activity_class'],axis = 1,inplace = True)
    all_bird_df.dropna(inplace = True)
    all_bird_df.reset_index(inplace = True)
    all_bird_df.drop(['index'],axis = 1,inplace = True)
    all_bird_df.to_csv(os.path.join(read_dir, 'all_bird_df_Y' + os.path.split(read_dir)[-1] +'_WL.csv'),index=False)
    
    labels = list(all_bird_df.drop_duplicates(subset=['label'],keep = 'first')['label'])
    print('labels_' + os.path.split(read_dir)[-1]+':')
    print(labels)
    return all_bird_df

def main(args):
    bird = args['bird']
    outdir = str(args['out_dir'])
    year = args['year']
    raw_path_o = os.path.join(OMIZU_PATH,'raw')
    raw_path_u = os.path.join(UMINEKO_PATH,'raw')

    if outdir == 'None':
        outdir = CSVWRITE_PATH
    
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

    all_bird_df = join_by_year(save_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bird', help="Bird kind to extract, either omizunagidoriu/umineko (O/U)", required=True)
    parser.add_argument('-y', '--year', help="Year to extract", required=True)
    parser.add_argument('-o', '--out-dir', help="Path to the output directory")

    args = vars(parser.parse_args())
    main(args)