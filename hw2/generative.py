import pandas as pd
import numpy as np
from numpy.linalg import inv
import sys

def readData(args):
    X_train = pd.read_csv(args[3])
    Y_train = pd.read_csv(args[4],header=None)
    X_test = pd.read_csv(args[5])
    return np.array(X_train), np.reshape(np.array(Y_train),(-1,1)), np.array(X_test)

def train(X_train, Y_train):
    trainSize = X_train.shape[0]
    findPos = np.where(Y_train == 1)
    findNeg = np.where(Y_train == 0)
    count1 = findPos[0].shape[0]
    count2 = findNeg[0].shape[0]
    mu1 = np.reshape(np.mean(X_train[findPos[0]], axis=0),(1,-1))
    mu2 = np.reshape(np.mean(X_train[findNeg[0]], axis=0),(1,-1))
    sigma1 = np.dot((X_train[findPos[0]] - mu1).T, X_train[findPos[0]] - mu1) / trainSize
    sigma2 = np.dot((X_train[findNeg[0]] - mu1).T, X_train[findNeg[0]] - mu1) / trainSize

    shared_sigma = (float(count1) / trainSize) * sigma1 + (float(count2) / trainSize) * sigma2
    return (mu1, mu2, shared_sigma, count1, count2)

def sigmoid(z):
    z = np.clip(z, -709, 709)
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def predict(X_test, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = inv(shared_sigma)
    w = np.dot((mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot(mu1, sigma_inverse), mu1.T) + (0.5) * np.dot(np.dot(mu2, sigma_inverse), mu2.T) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    a = a.T
    y = sigmoid(a)
    y[y>=0.5] = 1
    y[y<0.5] = 0
    return y

def outputFile(args,predictY):
    with open(args[6], 'w') as f:
        print('id,label', file=f)
        for (i, p) in enumerate(predictY):
            print('{},{}'.format(i+1, int(p[0])), file=f)

def main(args):
    X_train, Y_train, X_test = readData(args)
    data = np.concatenate((X_train,X_test), axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    X_train = (X_train-mean) / (std + 1e-100)
    X_test = (X_test-mean) / (std + 1e-100)
    
    mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train)
    predictY = predict(X_test, mu1, mu2, shared_sigma, N1, N2)
    outputFile(args,predictY)
    

if __name__ == '__main__':
	main(sys.argv)