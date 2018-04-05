import keras
import pandas as pd
import numpy as np
import sys
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import h5py
f = h5py.File('model.h5','r+')
del f['optimizer_weights']
f.close()
ATTR = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"]

def readData(trainPath, testPath):
    X_train = pd.read_csv(trainPath)
    X_test = pd.read_csv(testPath)
    trainX, trainY, testX = process(X_train, X_test)
    return trainX, trainY, testX

def process(X_train, X_test):
    needExpand = ["workclass","education","marital_status","occupation","relationship","race","sex","native_country"]
    squareAttr = ["fnlwgt","capital_gain","capital_loss","hours_per_week"]
    cubicAttr = ["capital_gain","hours_per_week","education_num"]
    # process trainY
    trainY = []
    dataY = X_train["income"]
    for i in range(len(dataY)):
        if (dataY[i] == " <=50K"):
            trainY.append(0)
        else:
            trainY.append(1)
    trainY = np.reshape(np.array(trainY),(-1,1))
    X_train = X_train.drop("income",axis=1)
    trainX = np.array([[]]*X_train.shape[0])
    testX = np.array([[]]*X_test.shape[0])
    for attr in list(X_train):
        if (attr in needExpand):
            expandAttrs = expandAttr(X_train[attr])
            trainX = np.concatenate((trainX,exapndData(X_train[attr],expandAttrs)), axis = 1)
            testX = np.concatenate((testX,exapndData(X_test[attr],expandAttrs)), axis = 1)
        else:
            if (attr == 'age'):
                trainX = np.concatenate((trainX,expandAge(X_train['age'])),axis = 1)
                testX = np.concatenate((testX,expandAge(X_test['age'])),axis = 1)
                continue
            tmp1 = np.reshape(np.array(X_train[attr]),(-1,1))
            tmp2 = np.reshape(np.array(X_test[attr]),(-1,1))
            tmpNormal1, mean, std = normalization(tmp1)
            tmpNormal2 = (tmp2-mean)/std
            trainX = np.concatenate((trainX,tmpNormal1), axis = 1)
            testX = np.concatenate((testX,tmpNormal2), axis = 1)
            
            if (attr in squareAttr):
                squareNormal1, mean, std = normalization(tmp1*tmp1)
                squareNormal2 = (tmp2*tmp2-mean)/std
                trainX = np.concatenate((trainX, squareNormal1), axis = 1)
                testX = np.concatenate((testX, squareNormal2), axis = 1)
                
            if (attr in cubicAttr):
                squareNormal1, mean, std = normalization(tmp1**3)
                squareNormal2 = (tmp2**3-mean)/std  
                trainX = np.concatenate((trainX, squareNormal1), axis = 1)
                testX = np.concatenate((testX, squareNormal2), axis = 1)
    return trainX, trainY, testX
def normal(trainX, testX):
    size = trainX.shape[0]
    data = np.concatenate((trainX, testX),axis = 0)
    maxNum = max(data)
    minNum = min(data)
    data = (data-minNum) / (maxNum-minNum)
    return data[:size,:], data[size:,:]

def expandAttr(data):
    expand = []
    for i in range(len(data)):
        if (data[i] not in expand):
            expand.append(data[i])
    return expand

def exapndData(data, expand):
    output = []
    for i in range(len(data)):
        output.append([])
        for j in range(len(expand)):
            if (data[i] == expand[j]):
                output[i].append(1)
            else:
                output[i].append(0)
    return np.array(output)

def expandAge(data):
    # 0~30 30~40 40~50 50~60 60~100
    output = []
    for i in range(len(data)):
        output.append([])
        for j in range(5):
            if (int(data[i]) <30 and int(data[i])>=0):
                output[i].append(1)
            else:
                output[i].append(0)
            if (int(data[i]) <40 and int(data[i])>=30):
                output[i].append(1)
            else:
                output[i].append(0)
            if (int(data[i]) <50 and int(data[i])>=40):
                output[i].append(1)
            else:
                output[i].append(0)
            if (int(data[i]) <60 and int(data[i])>=50):
                output[i].append(1)
            else:
                output[i].append(0)
            if (int(data[i]) >=60):
                output[i].append(1)
            else:
                output[i].append(0)
    return np.array(output)

def normalization(trainX):
    mean = np.mean(trainX, axis = 0)
    std = np.std(trainX, axis=0) + 1e-20
    return (trainX-mean)/std, mean, std 

def splitVal(X,Y,size):
    X_rand,Y_rand = shuffle(X,Y)
    X_train, Y_train = X_rand[int(X.shape[0]*size):,:], Y_rand[int(Y.shape[0]*size):]
    X_val, Y_val = X_rand[:int(X.shape[0]*size),:], Y_rand[:int(Y.shape[0]*size)]
    return X_train, Y_train, X_val, Y_val

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    X = X[randomList,:]
    Y = Y[randomList]
    return X, Y
def sigmoid(z):
    z = np.clip(z, -709, 709)
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def predict(y):
    y[y>=0.5] = 1
    y[y<0.5] = 0
    return y

def main(args):
    trainX, trainY, testX = readData(args[1],args[2])
    X_train, Y_train, X_val, Y_val = splitVal(trainX,trainY,0.05)
#    model = Sequential()
#    model.add(Dense(1024,input_dim=X_train.shape[1]))
#     model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.2))
    
#    model.add(Dense(512))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Dropout(0.2))
    
    
#    model.add(Dense(1, activation='sigmoid'))
#    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
#    model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy'])
#    callback = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
#    model.fit(X_train, Y_train, epochs=1000,validation_data=(X_val, Y_val), shuffle=True, batch_size=512, callbacks=[callback])
#     score = model.evaluate(X_val, Y_val)
#     print(score)
#    model.save('model.h5')
    model = load_model('model.h5')
    y = model.predict(testX)
    predictY = predict(y)
    
    with open(args[6], 'w') as f:
        print('id,label', file=f)
        for (i, p) in enumerate(predictY):
            print('{},{}'.format(i+1, int(p[0])), file=f)
    return predictY

if __name__ == '__main__':
	main(sys.argv)
