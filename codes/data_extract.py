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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
#from keras.utils import np_utils
#from keras.utils.vis_utils import model_to_dot
#from keras.callbacks import EarlyStopping
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
label_path_o = os.path.join(OMIZU_PATH,'labels')
print(label_path_o)
label_path_u = os.path.join(UMINEKO_PATH,'labels')
print(label_path_u)

label_omizu2018 = os.path.join(label_path_o,'Omizunagidori2018_labels_20230719_180351.xml')
label_omizu2019 = os.path.join(label_path_o,'Omizunagidori2019_labels_20230719_180404.xml')
label_omizu2020 = os.path.join(label_path_o,'Omizunagidori2020_labels_20230719_180417.xml')
label_omizu2021 = os.path.join(label_path_o,'Omizunagidori2021_labels_20230719_180425.xml')
label_omizu2022 = os.path.join(label_path_o,'Omizunagidori2022_labels_20230719_180438.xml')

label_paths_o = [label_omizu2018,label_omizu2019,label_omizu2020,label_omizu2021,label_omizu2022]

label_umineko2018 = os.path.join(label_path_u,'Umineko2018_labels_20230719_193014.xml')
label_umineko2019 = os.path.join(label_path_u,'Umineko2019_labels_20230719_193024.xml')
label_umineko2022 = os.path.join(label_path_u,'Umineko2022_labels_20230719_193039.xml')

label_paths_u = [label_umineko2018,label_umineko2019,label_umineko2022]

def make_labels(paths, label_wr_dir = '/home/bob/biodata/database/labels', fn_end = 17):
    
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
            
        print('created labels for >>> ',filename)
        
    return