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