# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:23:17 2017

@author: Esmeralda Chang
"""


from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM, BatchNormalization, TimeDistributed, Activation,Dense,Input,Dropout
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import mnist
from keras.layers import SimpleRNN
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import falldowndata_optical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import time


nb_classes = 2
K.set_image_dim_ordering('th')
(X_train, Y_train),(X_test, Y_test) = 光流與前處理.load_data()
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
#

TIME_STEPS = 3
INPUT_SIZE = 512
CELL_SIZE = 32
LR = 0.006
BATCH_SIZE = 20
EPOCH = 25

del model
input_features = Input(shape=(3,512), name='features')
input_normalized = BatchNormalization(name='normalization')(input_features)
input_dropout = Dropout(p=0.5)(input_normalized)
lstms_inputs = [input_dropout]

previous_layer = lstms_inputs[-1]
lstm = LSTM(128,input_shape=(3, 512),return_sequences=True,name='lsmt1')(previous_layer)
previous_layer = lstms_inputs[-1]
lstm = LSTM(128, name='lsmt2')(previous_layer)
lstms_inputs.append(lstm)
        
output_dropout = Dropout(p=0.5)(lstms_inputs[-1])
output = Dense(nb_classes, activation='softmax')(output_dropout)

model = Model(input=input_features, output=output)
rmsprop = RMSprop(lr=0.006)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                  batch_size=20,
                  validation_data=(X_test, Y_test),
                  verbose=1,
                  nb_epoch=30)


"""draw figure"""
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
