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
import sys

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
import logging


from load_utils import num_labels, time_change, setup_dir, find_xml_filenames, find_csv_filenames

warnings.filterwarnings('ignore')

BIODATA_PATH = os.environ['BIODATA_PATH']
OMIZU_PATH = os.environ["OMIZU_PATH"]
UMINEKO_PATH = os.environ["UMINEKO_PATH"]
CSVWRITE_PATH = os.environ['CSVWRITE_PATH']
LABELS_PATH = os.environ["LABELS_PATH"]
O_WRITE_PATH = os.environ['O_WRITE_PATH']
U_WRITE_PATH = os.environ['U_WRITE_PATH']

logger = logging.getLogger(__name__)

def separate_by_sensor(filename, save_folder, sensor='acc', time_format="%Y%m%d_%H:%M:%S.%f"):
    print(filename)
    data = pd.read_csv(filename, parse_dates=["timestamp"])
    data["timestamp"] = pd.to_datetime(data["timestamp"],format=time_format)
    
    #data['label'] = np.where(data['timestamp']=='NaN', 'unknown', data['label'])
    data.fillna('unlabeled',inplace=True)
    
    if sensor == 'acc':
        new_df = data.drop(['logger_id', 'latitude', 'longitude', 'gps_status', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'illumination', 'pressure', 'temperature','activity_class'],axis=1)
    else:
        new_df = data
        
    name = filename[-11:-4]
    name = name.replace('_','')
    name = name.replace('00','')
    save_name = os.path.join(save_folder,'data_w_timestamps',name+'_acc.csv')
    new_df.to_csv(save_name,index=False)
    #print('Saved to: ', save_name)
    logger.debug(f">> Done: Saved single bird CSV [path={save_name}, df={len(new_df)}]")
    
    #data_df = pd.read_csv(save_name, parse_dates=["timestamp"])
    data_df = new_df
    #l = []
    #for i in range(len(data)):
    #    l.append(data_df['timestamp'][i].replace('+00:00',''))
    #data_df['timestamp'] = l
    #data_df['timestamp'][0] = data_df['timestamp'][0]+'.000000'
    
    #data_df = pd.to_datetime(data_df["timestamp"],format=time_format)
    
    data_df = time_change(data_df)
    
    #data_df.to_csv(os.path.join(save_folder,name+'t_acc.csv'),index=False)
    return data_df
    
def join_by_year(read_dir):

    all_bird_lst = []
    for file in os.listdir(os.path.join(read_dir,'data_w_timestamps')):
        if file.endswith(".csv"):
            path = os.path.join(read_dir, 'data_w_timestamps', file)
            #print('added: ',path)
            logger.debug(f">> Added {path}, to year DataFrame")
            df = pd.read_csv(path,parse_dates=["timestamp"])
            all_bird_lst.append(df)
            #all_bird_df = all_bird_df._append(df, ignore_index=True)
            #all_bird_df = pd.concat([all_bird_df,df])

    all_bird_df = pd.concat(all_bird_lst)

    birds = list(all_bird_df.drop_duplicates(subset=['animal_tag'],keep = 'first')['animal_tag'])
    #print('birds included: ',birds)
    logger.debug(f">> Birds saved ={birds}")
    
    all_bird_df.to_csv(os.path.join(read_dir, 'all_bird_df.csv'),index=False)

    all_bird_df.dropna(inplace = True)
    all_bird_df.reset_index(inplace = True)
    all_bird_df.drop(['index'],axis = 1,inplace = True)
    all_bird_df.to_csv(os.path.join(read_dir, 'all_bird_df_Y' + os.path.split(read_dir)[-1] +'_WL.csv'),index=False)
    #print('Saved to: ',os.path.join(read_dir, 'all_bird_df_Y' + os.path.split(read_dir)[-1] +'_WL.csv'))
    logger.debug(f">> Done: Saved yearly CSV [path={os.path.join(read_dir, 'all_bird_df_Y' + os.path.split(read_dir)[-1] +'_WL.csv')}, df={len(all_bird_df)}]")
    
    labels_df = all_bird_df.drop_duplicates(subset=['label'],keep = 'first')['label']
    labels_l = list(labels_df)
    print('labels from' + os.path.split(read_dir)[-1]+':')
    print(labels_l)
    labels_df.to_csv(os.path.join(read_dir, 'label_df_Y' + os.path.split(read_dir)[-1] +'.csv'),index=False)
    #print('saved to: ', os.path.join(read_dir, 'label_df_Y' + os.path.split(read_dir)[-1] +'.csv'))
    logger.debug(f">> Done: Saved yearly labels CSV to ={os.path.join(read_dir, 'label_df_Y' + os.path.split(read_dir)[-1] +'.csv')}")
    
    return all_bird_df

def create_acc_files(all_raw_paths,all_save_folder,bird_n):

    all_bird_Y_df_l = []
    year_df_t_l = []

    for i in range(len(all_save_folder)):
        bird_df_t_l =[]
        for raw_path in all_raw_paths[i]:
            print('Acc sensor df from: ', raw_path)
            logger.debug(f">> Load Acc sensor df from:={raw_path}")
            bird_acc_df = separate_by_sensor(raw_path,all_save_folder[i])
            bird_df_t_l.append(bird_acc_df)
        year_df_t = pd.concat(bird_df_t_l)
        year_df_t_l.append(year_df_t)
        all_bird = join_by_year(all_save_folder[i])
        all_bird_Y_df_l.append(all_bird)

### Join all birds by year with Unix time

    all_t_df = pd.concat(year_df_t_l)
    #all_t_df.drop(['activity_class'], axis=1, inplace=True)
    all_t_df.dropna(inplace=True)
    all_t_df.reset_index(inplace = True)
    all_t_df.drop(['index'],axis = 1,inplace = True)
    #print('timechange dataframe: ')
    #print(all_t_df)

    all_df = pd.concat(all_bird_Y_df_l)
    labels_df = pd.DataFrame(all_df.drop_duplicates(subset=['label'],keep = 'first')['label'])
    labels_df.reset_index(inplace=True)
    labels_df.drop(['index'],axis=1,inplace=True)

    if bird_n=='omizunagidori':
        all_t_df.to_csv(os.path.join(O_WRITE_PATH,'Omizu_all_t_df.csv'), index = False)
        labels_df.to_csv(os.path.join(LABELS_PATH,'O_labels_df.csv'),index = False)
        logger.debug(f">> DONE: Saved labeled CSV [Path={os.path.join(O_WRITE_PATH,'Omizu_all_t_df.csv')}, df={len(all_t_df)}]")
        logger.debug(f">> DONE: Saved labels CSV to = {os.path.join(LABELS_PATH,'O_labels_df.csv')}")
    elif bird_n=='umineko':
        all_t_df.to_csv(os.path.join(U_WRITE_PATH,'Umineko_all_t_df.csv'), index = False)
        labels_df.to_csv(os.path.join(LABELS_PATH,'U_labels_df.csv'),index = False)
        logger.debug(f">> DONE: Saved labeled CSV [Path={os.path.join(O_WRITE_PATH,'Umineko_all_t_df.csv')}, df={len(all_t_df)}]")
        logger.debug(f">> DONE: Saved labels CSV to = {os.path.join(LABELS_PATH,'U_labels_df.csv')}")

def list_of_strings(arg):
    return arg.split(',')

def main(args):
    bird = args['bird']
    outdir = str(args['out_dir'])
    years = args['year']
    raw_path_o = os.path.join(OMIZU_PATH,'raw')
    raw_path_u = os.path.join(UMINEKO_PATH,'raw')

    if outdir == 'None':
        outdir = str(CSVWRITE_PATH)

    all_save_folders = []
    all_raw_paths = []

    if bird == 'omizunagidori' or bird == 'O':
        for year in years:
            save_folder = os.path.join(outdir, 'omizunagidori', year)
            raw_paths = find_csv_filenames(raw_path_o, suffix = ".csv", year = year)
            all_save_folders.append(save_folder)
            all_raw_paths.append(raw_paths)
        bird_n = 'omizunagidori'
    elif bird == 'umineko' or bird == 'U':
        for year in years:
            save_folder = os.path.join(outdir, 'umineko', year)
            raw_paths = find_csv_filenames(raw_path_u, suffix=".csv", year = year)
            all_save_folders.append(save_folder)
            all_raw_paths.append(raw_paths)
        bird_n = 'umineko'
    
    create_acc_files(all_raw_paths,all_save_folders,bird_n)

    #print("year list: ", years)
    logger.debug(f">> DONE: years extracted = {years} for {bird_n}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bird', help="Bird kind to extract, either omizunagidoriu/umineko (O/U)", required=True)
    parser.add_argument('-y', '--year', help="Year to extract", required=True, type=list_of_strings)
    parser.add_argument('-o', '--out-dir', help="Path to the output directory")

    args = vars(parser.parse_args())
    main(args)