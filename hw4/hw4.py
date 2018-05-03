import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
import sys

class cluster():
    def __int__(self):
        self._train_data = ""
        self._val_data = ""
        
    def read_data(self, train_data):
        train_num = 130000
        X = np.load(train_data).astype("float32") / 255
        self._all_data = np.reshape(X, (len(X), -1))
        self._train_data = X[:train_num]
        self._val_data = X[train_num:]
    
    def build_model(self):
        input_img  = Input(shape=(784, ))
        encoded = Dense(1024,activation='relu')(self._input_img)
        encoded = Dense(512,activation='relu')(encoded)
        encoded = Dense(512,activation='relu')(encoded)
        encoded = Dense(256,activation='relu')(encoded)
        encoded = Dense(256,activation='relu')(encoded)
        encoded = Dense(64,activation='relu')(encoded)
        encoded = Dense(64,activation='relu')(encoded)

        decoded = Dense(64,activation='relu')(encoded)
        decoded = Dense(128,activation='relu')(decoded)
        decoded = Dense(128,activation='relu')(decoded)
        decoded = Dense(256,activation='relu')(decoded)
        decoded = Dense(512,activation='relu')(decoded)
        decoded = Dense(512,activation='relu')(decoded)
        decoded = Dense(784,activation='relu')(decoded)
        encoder = Model(input=input_img, output=encoded)
        autoencoder = Model(input=input_img, output=decoded)
        self._autoencoder = autoencoder
        self._encoder = encoder
    
    def compile_model(self):
        adam = Adam(lr=5e-4)
        self._autoencoder.compile(optimizer=adam, loss='mse')
    
    def summary(self):
        self._autoencoder.summary()
        
    def fit(self):
        self._autoencoder.fit(self._train_data, self._train_data, epochs=50, batch_size=256, shuffle=True, validation_data=(self._val_data, self._val_data))
        
    def load(self, autoencoder_model, encoder_model):
        self._autoencoder = load_model(autoencoder_model)
        self._encoder = load_model(encoder_model)
    
    def predict_kmeans(self):
        encoded_imgs = self._encoder.predict(self._all_data)
        encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],-1)
        self._kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
    
    def write_output(self, test_data,  output):
        f = pd.read_csv(test_data)
        IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']),np.array(f['image2_index'])
        o = open(output,"w")
        o.write("ID,Ans\n")
        for idx, i1, i2 in zip(IDs, idx1, idx2):
            p1 = self._kmeans.labels_[i1]
            p2 = self._kmeans.labels_[i2]
            if (p1 == p2):
                pred = 1
            else:
                pred = 0
            o.write("{},{}\n".format(idx, pred))
        o.close()

def main(argvs):
	image_cluster = cluster()
	image_cluster.read_data(argvs[1])
	image_cluster.load("autoencoder.h5", "encoder.h5")
	image_cluster.predict_kmeans()
	image_cluster.write_output(argvs[2],argvs[3])

if __name__ == '__main__':
    main(sys.argv)