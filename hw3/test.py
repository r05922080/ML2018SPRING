import keras
import pandas as pd
import numpy as np
from keras.models import load_model
import sys
BATCHSIZE = 128;
EPOCH = 300;

def readData(label, data_path):
    if (label == "train"):
        train = pd.read_csv(data_path)
        train_Y = train["label"]
        train = train["feature"]
        tmp = []
        for i in range(train.shape[0]):
            tmp.append(train[i].split(" "))
        train_X = np.array(tmp).astype('float') / 255 
        train_X = np.reshape(train_X, (-1,48,48,1))
        train_Y = to_categorical(train_Y, num_classes=7)
        return train_X, train_Y
    elif (label == "test"):
        test = pd.read_csv(data_path)
        test_X = test['feature']
        
        tmp2 = []
        for i in range(test_X.shape[0]):
            tmp2.append(test_X[i].split(" "))
        test_X = np.array(tmp2).astype('float') / 255 
        test_X = np.reshape(test_X,(-1,48,48,1))
    return test_X

def build_model():
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

    model.summary()
    return model

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    X = X[randomList,:,:,:]
    Y = Y[randomList,:]
    return X, Y

def splitVal(X,Y,size):
    X_rand,Y_rand = shuffle(X,Y)
    X_train, Y_train = X_rand[int(X.shape[0]*size):,:], Y_rand[int(Y.shape[0]*size):]
    X_val, Y_val = X_rand[:int(X.shape[0]*size),:], Y_rand[:int(Y.shape[0]*size)]
    return X_train, Y_train, X_val, Y_val


def writeOutput(predictY, file_path):
    with open(file_path, 'w') as f:
        print('id,label', file=f)
        for (i, p) in enumerate(predictY):
            print('{},{}'.format(i, int(p)), file=f)

def main():
    train_X, train_Y = readData("train")
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    train_X = (train_X-mean) / (std + 1e-20)
    X_train, Y_train, X_val, Y_val = splitVal(train_X, train_Y, 0.1)
  
    X_train = np.concatenate((X_train, X_train[:,:,::-1]), axis = 0)
    Y_train = np.concatenate((Y_train, Y_train), axis = 0)
    emotion_classifier = build_model()
    callbacks = []
    callbacks.append(ModelCheckpoint('model.h5', monitor='val_acc', save_best_only=True, period=1))
    callbacks.append(EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto'))
    emotion_classifier.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=BATCHSIZE), 
            steps_per_epoch=5*len(X_train)//BATCHSIZE,
            epochs=EPOCH,
            validation_data=(X_val, Y_val),
            callbacks=callbacks
    )
    return mean, std;

def test(argv):
    test_X = readData("test",argv[1])
    tmp = np.load("normalize.npy")
    mean = tmp[0]
    std = tmp[1]
    X_test = (test_X-mean) / (std+1e-20)
    model = load_model("model.h5")
    y = model.predict(X_test)
    predictY = []
    for i in range(y.shape[0]):
        predictY.append(np.argmax(y[i]))
    writeOutput(predictY, argv[2])

if __name__ == '__main__':
    test(sys.argv)
