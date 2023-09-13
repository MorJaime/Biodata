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

def make_labels(paths, label_wr_dir = LABELS_PATH, fn_end = 17):
    
    for path in paths:
        label_path = path
        label_dir_p, label_fn = os.path.split(label_path)
        wr_fn = label_fn[:fn_end]
        tree = ET.parse(label_path)
        root = tree.getroot()
        filename = os.path.join(label_wr_dir,wr_fn+'_labels.csv')
        
        with open(filename,"w") as f:            
            csv_writer = csv.writer(f)
            header = ["event_type","start", "end"]
            csv_writer.writerow(header)
            for labellist in root.iter("labellist"):
                timestampStart = labellist[1].text
                timestampStart = timestampStart.replace('-','')
                timestampStart = timestampStart.replace('T',' ')      
                timestampStart = timestampStart.replace('Z','')
                timestampEnd = labellist[2].text
                timestampEnd = timestampEnd.replace('-','')
                timestampEnd = timestampEnd.replace('T',' ')
                timestampEnd = timestampEnd.replace('Z','')

                row = [labellist[0].text, labellist[1].text, labellist[2].text]
                row = [labellist[0].text,timestampStart,timestampEnd]
                csv_writer.writerow(row)
            
        #print('created labels for >>> ',filename)
        logger.info(f">>> DONE: Created labels in: {filename}")
        
    return

def main(args):
    outdir = str(args['out_dir'])
    label_path_o = os.path.join(OMIZU_PATH,'labels')
    label_path_u = os.path.join(UMINEKO_PATH,'labels')
    
    if outdir == "None":
        wrdir = LABELS_PATH
    else:
        wrdir = outdir
    
    label_paths_o = find_xml_filenames(label_path_o)
    label_paths_u = find_xml_filenames(label_path_u)

    make_labels(label_paths_o, label_wr_dir = os.path.join(wrdir,'omizunagidori'), fn_end = 17)
    make_labels(label_paths_u, label_wr_dir = os.path.join(wrdir,'umineko'), fn_end = 11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', help="Path to the output directory")

    args = vars(parser.parse_args())
    main(args)