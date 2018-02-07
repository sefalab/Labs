#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as pl
from sklearn.ensemble import RandomForestClassifier

def x_2_feature(x):
    """
    returns a d-dimensional feature vector v that divides the image into 25 segments
    """
    x1 = np.reshape(x,(28,28))
    v = np.zeros((5,5), dtype=np.float)
    for i in range(0,5):
        row = i*5
        row1 = (i+1)*5
        for j in range(0,5):
            col = j*5
            col1 = (j+1)*5
            v[i][j] = np.mean(x1[row:row1,col:col1])
    v = np.reshape(v,25)
    
    return v

def act(s):
    """
    activation function
    """
    u = 1.0 / (1.0 + np.exp(-s))
    return u

def ann_classify(x, theta):
    """
    classifies x according ffnn architecture
    """
    L = len(theta)
    s = dict()
    u = dict()
    s[0] = x.copy()
    u[0] = s[0].copy()
    for l in np.arange(1, L+1): # apply forward propagation
        s[l] = np.dot(u[l-1], theta[l-1])
        u[l] = act(s[l])
        if l != L:
            u[l][0] = 1.0
    return u[l]


def digit(x_classify):
    x = x_2_feature(x_classify)
    x = x /255
    theta = dict()
    theta[0] = np.loadtxt('data1.dat', delimiter=' ');
    theta[1] = np.loadtxt('data2.dat', delimiter=' ')
    h = ann_classify(x, theta)
    i= np.argsort(h)
    label= i[9]
    return label


def capture(name):
#    camera_port = 0
#    camera = cv2.VideoCapture(camera_port)
#    time.sleep(0.1)  # If you don't wait, the image will be dark
#    return_value, image = camera.read()
#    cv2.imwrite("opencv5.png", image)
#    del(camera)  # so that others can use the camera as soon as possible
    image = cv2.imread(name)
    gray_im=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(gray_im, (28, 28))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(name, gray_image)
       
    I = pl.imread(name)*255
    w=np.reshape(I, 784)
    for i in np.arange(w.shape[0]):
        if w[i]>=100:
            w[i]=0
    return w

name= 'Num_[3].png'
w1=capture(name)

n1= digit(w1[np.newaxis,:])

print(n1)

def load_data():
    data= pd.read_csv('training.txt', header=None)
    
    data_x_training =data[:24000]
    data_y_training=np.copy(data_x_training[784])
    del data_x_training[784]
    
    data_x_val =data[24000:36000]
    data_y_val=np.copy(data_x_val[784])
    del data_x_val[784]
    
    data_x_test =data[36000:]
    data_y_test=np.copy(data_x_test[784])
    del data_x_test[784]
    
#    eg=data_x_training.iloc[8,:].values
#    eg=eg.reshape(28,28).astype('uint8')
#    plt.imshow(eg)
#
#    data_y_training[8]

    return data_x_training,data_y_training,data_x_val,data_y_val,data_x_test,data_y_test

data_x_training,data_y_training,data_x_val,data_y_val,data_x_test,data_y_test=load_data()
rf=RandomForestClassifier(n_estimators=100)
rf.fit(data_x_training,data_y_training)
def svm(data_x_val):
    
    
    return rf.predict(data_x_val)

print(svm(w1))
#w2=capture()  
#n2= digit(w2[np.newaxis,:])
#print(n2) 
n2=2

print('sum:')
print(  (n1+n2))         
#data= pd.read_csv('training.txt', header=None)   
#data_ =np.array(data.iloc[:60000])
#data_y_training=np.copy(data_[:,784])
#data_x_training= np.copy(data_[:,0:784]) 