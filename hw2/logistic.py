import pandas as pd
import numpy as np
import sys
ATTR = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"]

def readData(trainPath, testPath):
    X_train = pd.read_csv(trainPath)
    X_test = pd.read_csv(testPath)
    trainX, trainY, testX = process(X_train, X_test)
    return trainX, trainY, testX

def process(X_train, X_test):
    needExpand = ["workclass","education","marital_status","occupation","relationship","race","sex","native_country"]
#     removeAttr = ["capital_gain","capital_loss","relationship"]
    removeAttr = []
    
    # process trainY
    trainY = []
    dataY = X_train["income"]
    for i in range(len(dataY)):
        if (dataY[i] == " <=50K"):
            trainY.append(0)
        else:
            trainY.append(1)
    trainY = np.reshape(np.array(trainY),(-1,1))
    # process trainX: expand the attributes
    X_train = X_train.drop("income",axis=1)
    trainX = np.array([[]]*X_train.shape[0])
    testX = np.array([[]]*X_test.shape[0])
    for attr in list(X_train):
        if (attr in removeAttr):
            continue
        if (attr in needExpand):
            expandAttrs = expandAttr(X_train[attr])
            trainX = np.concatenate((trainX,exapndData(X_train[attr],expandAttrs)), axis = 1)
            testX = np.concatenate((testX,exapndData(X_test[attr],expandAttrs)), axis = 1)
        else:
            tmp1 = np.reshape(np.array(X_train[attr]),(-1,1))
            tmp2 = np.reshape(np.array(X_test[attr]),(-1,1))
            tmp1, mean, std = normalization(tmp1)
            tmp2 = (tmp2-mean)/std
            trainX = np.concatenate((trainX,tmp1), axis = 1)
            testX = np.concatenate((testX,tmp2), axis = 1)
    return trainX, trainY, testX

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

def train(X, Y, X_val, Y_val ,EPOCH,LEARNINGRATE):

    Y = np.reshape(Y,(-1,1))
    Y_val = np.reshape(Y_val,(-1,1))
    Weight = np.zeros((X.shape[1],1))          # (108,1)
    Bias = 0.0
    AdagradW = np.zeros((X.shape[1],1))        # (108,1)
    AdagradB = 0.0
    lamda = 0.0
    
    LOSS = []
    for i in range(EPOCH):
        
        z = np.dot(X,Weight) + Bias
        y = sigmoid(z)
        error = Y - y
#         print(np.sum(error))
        z_val = np.dot(X_val,Weight) + Bias
        y_val = sigmoid(z_val)
        val_error = Y_val - y_val
        
        gradientB = -np.sum(error)*1
        gradientW = -np.dot(X.T,error)
     
        AdagradB += gradientB ** 2 
        AdagradW += gradientW ** 2
        
        Bias = Bias - LEARNINGRATE / np.sqrt(AdagradB) * gradientB
        Weight = Weight - LEARNINGRATE / np.sqrt(AdagradW) * gradientW
        
        cross_entropy = -np.mean((Y*np.log(y)) + (1 - Y)*np.log(1 - y))
        cross_entropy_val = -np.mean((Y_val*np.log(y_val)) + (1 - Y_val)*np.log(1 - y_val))
        LOSS.append(cross_entropy)
        if (i+1) % 1000 == 0:
            train_acc = accuracy(Y, y)
            try:
                val_acc = accuracy(Y_val, y_val)
            except:
                val_acc = "nan"
            print("epoch:",i+1, " train Loss:",cross_entropy, " val loss:",cross_entropy_val, " train acc:",train_acc, " val acc:",val_acc)
            
    return Weight, Bias, cross_entropy, LOSS

def normalization(trainX):
    mean = np.mean(trainX, axis = 0)
    std = np.std(trainX, axis=0) + 1e-20
    return (trainX-mean)/std, mean, std  


def accuracy(Y,y):
    y[y>=0.5] = 1
    y[y<0.5] = 0
    acc = np.where(Y-y == 0)
    return (float)(acc[0].shape[0]) / y.shape[0]

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


def predict(test, Weight, Bias):
    y = sigmoid(np.dot(test,Weight) + Bias)
    y[y>=0.5] = 1
    y[y<0.5] = 0
    return y

def main(args):
	trainX, trainY, testX = readData(args[1],args[2])
	X_train, Y_train, X_val, Y_val = splitVal(trainX,trainY,0.0)
	EPOCH = 100000
	LEARNINGRATE = 1
	# Weight, Bias, Loss, LOSS, acc = train(X_train, Y_train, X_val, Y_val ,EPOCH, LEARNINGRATE)
	Weight = np.loadtxt("weight.txt")
	Weight = np.reshape(Weight,(-1,1))
	Bias = np.loadtxt("bias.txt")

	predictY = predict(testX, Weight, Bias)

	with open(args[6], 'w') as f:
	    print('id,label', file=f)
	    for (i, p) in enumerate(predictY):
	        print('{},{}'.format(i+1, int(p[0])), file=f)

if __name__ == '__main__':
	main(sys.argv)
