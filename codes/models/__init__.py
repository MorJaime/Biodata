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

print(">>> Done: Succesfully imported Utils module")