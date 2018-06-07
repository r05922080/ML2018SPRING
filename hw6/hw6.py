import sys
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping

def read_test(path):
    data = np.array(pd.read_csv(path, encoding='big5'))
    return data

def MF_model():
    U_input = Input(shape=[1])
    M_input = Input(shape=[1])
    
    U_embedding = Embedding(8192, 16, embeddings_initializer='random_normal')(U_input)
    U_embedding = Flatten()(U_embedding)

    M_embedding = Embedding(4096, 16, embeddings_initializer='random_normal')(M_input)
    M_embedding = Flatten()(M_embedding)

    output_data = Dot(axes=1)([U_embedding, M_embedding])

    model = Model([U_input, M_input], output_data)
    model.summary()
    return model
    
def RMSE(y, pred):
    pred = K.clip(pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y - pred))))

def write_file(predict, path):
    with open(path, 'w') as f:
        print('TestDataID,Rating', file=f)
        for (i, p) in enumerate(predict):
            print('{},{}'.format(i+1, p), file=f)

def test(data, model, save_path):
    predict = model.predict([data[:,1], data[:,2]], batch_size=512)
    predict = np.clip(predict, 1, 5).reshape(-1)
    write_file(predict, save_path)

def main(argv):
    model = MF_model()
    model.load_weights("model.h5")
    model.compile(loss='mse', optimizer='adam', metrics=[RMSE])
    test_data = read_test(argv[1])
    test(test_data, model, argv[2])
    
if __name__ == '__main__':
	main(sys.argv)