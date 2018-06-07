import numpy as np
import pickle
import sys
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, GRU, LeakyReLU
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

def read_train_data(file_path):
    train_x = []
    train_y = []
    file = open(file_path ,"r")
    data = file.read()
    for line_data in data.split("\n")[:-1]:
        l = line_data.split(' +++$+++ ')
        train_y.append(int(l[0]))
        train_x.append(l[1])
    return np.array(train_x), np.array(train_y)

def build_token(data):
    tokenizer = Tokenizer(num_words = 20000, filters='\n')
    tokenizer.fit_on_texts(data)
    pickle.dump(tokenizer, open('token.pk', 'wb'))
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen=32)
    return data


def main(argv):
    label_x, label_y = read_train_data(argv[1])
    train_X = build_token(label_x)
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystopping = EarlyStopping(monitor='val_acc', patience=2, mode='max')
    checkpoint = ModelCheckpoint(filepath="model.h5", save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')
    model.fit(train_X, label_y, batch_size = 512, epochs = 10 , validation_split=0.1, callbacks=[checkpoint, earlystopping]) #20

if __name__ == '__main__':
	main(sys.argv)