import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, normalize
from sklearn.preprocessing import MinMaxScaler, Normalizer

def get_activity_index(act_dict):
    labels = []
    indices = []
    for k,v in act_dict.items():
        indices.append(k)
        labels.append(v)
    return labels, indices

def plot_confusion_matrix(cm, fig, cm_plt, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    simg = cm_plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cm_plt.set_title(title)
    tick_marks = np.arange(len(classes))
    cm_plt.set_xticks(tick_marks)
    cm_plt.set_yticks(tick_marks)
    cm_plt.set_xticklabels(classes)
    cm_plt.set_yticklabels(classes)
    fig.colorbar(simg)
    
    
    plt.setp(cm_plt.get_xticklabels(), rotation=45, horizontalalignment='right')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    cm_plt.set_ylabel('True label')
    cm_plt.set_xlabel('Predicted label')
    #cm_plt.tight_layout()

def get_Train_Test(inn_df,shuffle=False,trn=1000):
    
    in_df = inn_df.dropna()

    j = in_df[['job']]
    j['q'] = in_df.duplicated(['job'],keep = 'first')
    job_df = j[j['q'] == False]
    job_l = [int(i) for i in list(job_df['job'])]           
    max_len = 0

    for i in job_l:
        c_len = len(in_df[in_df['job'] == i])
        
        if c_len > max_len:
            print(i,' : ',c_len,' > ',max_len,' ? ',c_len > max_len)
            max_len = c_len
        print('job: ',i,'lenght: ',c_len)
    reqlen = int((int(max_len/512)+1)*512)
    print('reqlen: ',reqlen)

    features = in_df.columns[1:-3]
    X = []
    Y = []
    times = []
    
    min_max_scaler = MinMaxScaler()
    normalizer = Normalizer()

    for i in job_l:
        x = np.array(in_df[in_df['job'] == i][features].values)
        #print('x.shape',x.shape)
        xz = np.zeros((reqlen-len(x),len(features)))
        xs = normalizer.fit_transform(x)
        sx = np.concatenate((x,xz))
    
        y = np.array(in_df[in_df['job'] == i]['l_val'].values)
        z = np.array([10]*(reqlen-len(y)))
        sy = np.concatenate((y,z))
    
        t = np.array(in_df[in_df['job'] == i]['time'].values)
        st = np.concatenate((t,z))
    
        X.append([sx])
        Y.append([sy])
        times.append(st)
        
    if shuffle:
        p = list(zip(X,Y))
        random.shuffle(p)
        X, Y = zip(*p)
        X = list(X)
        Y = list(Y)

    X = np.array(X)
    Y = np.array(Y)

    times = np.array(times)

    print('X(shape): ',X.shape)
    print('Y(shape): ',Y.shape)
    print('times(shape): ',times.shape)
    
    selector=trn
    
    if trn == 1000:
        selector = random.randint(0,X.shape[0])
        testX = np.array([X[selector]])
        testY = np.array([Y[selector]])
        testtimes=times[selector]
        trainX = np.delete(X,selector,0)
        trainY = np.delete(Y,selector,0)
    else:
        testX = np.array([X[trn]])
        testY = np.array([Y[trn]])
        testtimes=times[trn]
        trainX = np.delete(X,trn,0)
        trainY = np.delete(Y,trn,0)
    
    return trainX,testX,trainY,testY,testtimes,selector

def timeseries_standardize(trainX, testX):

    ss = preprocessing.StandardScaler()
    trainXshape = trainX.shape 
    trainX = trainX.reshape(-1, trainX.shape[2]) 
    ss.fit(trainX)
    trainX = ss.transform(trainX) 
    trainX = trainX.reshape(trainXshape) 

    testXshape = testX.shape
    testX = testX.reshape(-1, testX.shape[2]) 
    testX = ss.transform(testX) 
    testX = testX.reshape(testXshape) 
    
    return trainX, testX