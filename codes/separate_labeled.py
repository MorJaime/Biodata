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

def divide_years(acc_df,write_folder,year_times):
    
    new_df = acc_df
    new_df['year']='0'
    years = list(year_times.keys())

    for year in years:
        print(year_times[year][0])
        year_df = new_df[new_df['timestamp']>year_times[year][0]]
        year_df = year_df[year_df['timestamp']<year_times[year][1]]
        new_df['year'] = np.where(new_df['timestamp'].between(year_times[year][0],year_times[year][1]), year, new_df['year'])
        year_df = new_df[new_df['year']==year]
        year_df.to_csv(os.path.join(write_folder,year,'l_'+year+'_acc.csv'),index = False)

def divide_birds(read_folder,year_times):
    
    years = list(year_times.keys())

    for year in years:
        year_df = pd.read_csv(os.path.join(read_folder,year,'l_'+year+'_acc.csv'))
        birds_df = pd.DataFrame(year_df.drop_duplicates(subset=['animal_tag'],keep = 'first')['animal_tag'])
        birds = list(birds_df['animal_tag'])
        print('year: ',year)
        for bird in birds:
            print('bird: '+bird)
            new_df = year_df[year_df['animal_tag']==bird]
            new_df.to_csv(os.path.join(read_folder,year,'labeled',bird+'_l_acc.csv'),index=False)

def main(args):
    bird = args['bird']



    if bird == 'omizunagidori' or bird == 'O':
        labels_all_df = pd.read_csv(os.path.join(LABELS_PATH,'O_labels_df.csv'))
        year_times = {'2018':[1514768460000,1546304460000],'2019':[1546304460000,1577840460000],
                        '2020':[1577840460000,1609462860000],'2021':[1609462860000,1640998860000],'2022':[1640998860000,1672534860000]}
        acc_path = str(O_WRITE_PATH)
        acc_fn = 'omizu_all_t_df.csv'
        
    elif bird == 'umineko' or bird == 'U':
        labels_all_df = pd.read_csv(os.path.join(LABELS_PATH,'U_labels_df.csv'))
        year_times = {'2018':[1514768460000,1546304460000],'2019':[1546304460000,1577840460000],'2022':[1640998860000,1672534860000]}
        acc_path = str(U_WRITE_PATH)
        acc_fn = 'umineko_all_t_df.csv'

    all_t_df = pd.read_csv(os.path.join(acc_path,acc_fn))
    lab_all_df = num_labels(all_t_df,labels_all_df)
    divide_years(lab_all_df,acc_path,year_times)
    divide_birds(acc_path,year_times)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bird', help="Bird kind to label, either omizunagidoriu/umineko (O/U)", required=True)

    args = vars(parser.parse_args())
    main(args)