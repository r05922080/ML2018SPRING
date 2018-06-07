import numpy as np
import pickle as pk
import sys, argparse, os, csv
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, GRU, LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model():
    model = Sequential()
    model.add(Embedding(20000+1, 1024, input_length=32, embeddings_initializer=keras.initializers.random_normal(stddev=1.0))) #256
    model.add(Bidirectional(GRU(512, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)))
    model.add(Bidirectional(GRU(32, dropout=0.4, recurrent_dropout=0.2)))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    return model

def read_test_data(file_path):
    test=[]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            if idx != 0:
                line = lines[idx].strip().split(',',1)
                test.append(line[1])
    return test


def build_test(x_test):
    with open('token.pk', 'rb') as f:
        tokenizer = pk.load(f)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=32)
    return x_test

def write_file(predict, file_name):
    with open(file_name, 'w') as f:
        print('id,label', file=f)
        for (i, p) in enumerate(predict):
            if (p[0] >= 0.5):
                print('{},{}'.format(i, 1), file=f)
            else:
                print('{},{}'.format(i, 0), file=f)


def main(argv):
	model = build_model()
	model.load_weights("model.h5")
	test = read_test_data(argv[1])
	test = build_test(test)
	predict = model.predict(test, batch_size = 512)
	write_file(predict, argv[2])

if __name__ == '__main__':
	main(sys.argv)