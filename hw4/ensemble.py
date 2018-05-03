import keras
import pandas as pd
import numpy as np
from keras.models import Sequential,load_model, Model
from keras.layers import Conv2D, Dense, Dropout, Activation, MaxPooling2D,ZeroPadding2D,AveragePooling2D,Input,Flatten, Average, concatenate
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import optimizers
from keras.initializers import random_normal
from keras.utils import plot_model
import gc

class ensemble_model():
    def __init__(self):
        self._mean = 0
        self._std = 1
        
    def read_data(self, label, data_path):
        if (label == "train"):
            train = pd.read_csv(data_path)
            train_Y = train["label"]
            train = train["feature"]
            tmp = []
            for i in range(train.shape[0]):
                tmp.append(train[i].split(" "))
            train_X = np.array(tmp).astype('float') / 255 
            train_X = np.reshape(train_X, (-1,48,48,1)).astype('float')
            train_Y = to_categorical(train_Y, num_classes=7)
            self._train_X = train_X
            self._train_Y = train_Y

        elif (label == "test"):
            test = pd.read_csv(data_path)
            test_X = test['feature']

            tmp2 = []
            for i in range(test_X.shape[0]):
                tmp2.append(test_X[i].split(" "))
            test_X = np.array(tmp2).astype('float') / 255 
            test_X = np.reshape(test_X,(-1,48,48,1)).astype('float')
            self._test_X = test_X
    def normalize(self, path):
        tmp = np.load(path)
        self._mean = tmp[0]
        self._std = tmp[1]
        self._test_X = (self._test_X - self._mean) / self._std
        
    def build_model1(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(48,48,1),activation='relu'))
        model.add(LeakyReLU(alpha=0.025))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))

        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.1))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))

        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=0.025))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.4))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self._model1 = model

    def build_model2(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(48,48,1), padding='same'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.35))

        model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        self._model2 = model
    def load_model(self, model_path1, model_path2):
        self._model1.load_weights(model_path1)
        self._model2.load_weights(model_path2)
        
    def predict(self, label):
        if (label == 'test'):
            y1 = self._model1.predict(self._test_X)
            y2 = self._model2.predict(self._test_X)
            y = (y1 + y2) / 2
        elif (label == 'train'):
            y1 = self._model1.predict(self._train_X)
            y2 = self._model2.predict(self._train_X)
            y = (y1 + y2) / 2
        self._predict_Y = []
        for i in range(y.shape[0]):
            self._predict_Y.append(np.argmax(y[i]))

    def write_output(self, path):
        with open(path, 'w') as f:
            print('id,label', file=f)
            for (i, p) in enumerate(self._predict_Y):
                print('{},{}'.format(i, int(p)), file=f)

        
ensemble = ensemble_model()
ensemble.read_data("test", "../../hw3/dataset/test.csv")
ensemble.normalize("normalize.npy")
ensemble.build_model1()
ensemble.build_model2()
ensemble.load_model("model/ensemble/model1_weight.h5","model/ensemble/model2_weight.h5")
ensemble.predict("test")
ensemble.write_output("aaa.csv")