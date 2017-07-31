# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 16:32:34 2017

@author: Esmeralda Chang
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from os import listdir   
from keras.models import load_model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D

def create_model():
    model = Sequential()
    model.add(Convolution3D(16,3, 3, 3,
                            border_mode='valid',
                            input_shape=(3,48,64,18),
                            activation='relu'))
    
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Convolution3D(32,3, 3, 3,
                            border_mode='valid',
                            activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.compile(loss='mae',optimizer="sgd")
    return model
    
def cnn_frame(model,rgbs):
    rgbs = rgbs.reshape(3,3,48,64,18)
    Flatten = model.predict(rgbs)  
    return Flatten


def opticalframe(model,vid):
    cap = cv2.VideoCapture(vid)
    ret, first_frame = cap.read()
    prvs = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[...,1] = 255
    idx = 0
    frames = []
    rgbs = []
    img_rows,img_cols=48,64
    
    while(1):
        ret, next_frame = cap.read()
        if next_frame is None:
            break;
        next = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        #HSV(消除雜訊)
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,2] = np.minimum(v*4, 255)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        rgb=cv2.resize(rgb,(img_cols,img_rows),interpolation=cv2.INTER_AREA)
        rgbs.append(rgb)
        idx = idx +1
        prvs = next
    rgbs = np.array(rgbs)
    frames = cnn_frame(model,rgbs)
    cap.release()
    cv2.destroyAllWindows()
    return frames

def load_data():
    video_rows, video_cols = 54,512
    img_rows,img_cols=48,64
    Y_train=[]
    X_train=[]
    ##Train
    path = '\\訓練資料\\'
    i = 0
    v_idx=0
    
    cnnmodel = load_model('\\cnn.h5')
    while i < 2:
        dir = '{0}{1}{2}'.format(path, i,'\\')
        listing = os.listdir(dir)
        vid = listing[0]
        for vid in listing:
            vid = dir+vid
            frames = []
            frames = opticalframe(cnnmodel,vid)
            X_train.append(frames)
            Y_train.append(i)
            v_idx = v_idx + 1
        i += 1   
    X_train = np.array(X_train) 
    X_train = X_train.astype('float64')

    

##Test
    X_test=[]
    Y_test=[]
    path = '\\測試資料\\'
    i = 0
    v_idx=0
    while i < 2:
        dir = '{0}{1}{2}'.format(path, i,'\\')
        listing = os.listdir(dir)
        for vid in listing:
            vid = dir+vid
            frames = []
            frames = opticalframe(cnnmodel,vid)
            X_test.append(frames)
            Y_test.append(i)
            v_idx = v_idx + 1
        i += 1   
    X_test = np.array(X_test)
    X_test = X_test.astype('float64')
    return (X_train, Y_train),(X_test, Y_test)
