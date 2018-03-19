import pandas as pd
import numpy as np
import sys

FILEPATH = "../dataset/train.csv"
ATTR = ["AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","PM2.5","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]

def readData():
    # read csv
    data = pd.read_csv(FILEPATH, sep=",", encoding = "ISO-8859-1", skiprows=1,  names=['date', 'place', 'kind', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'])
    data = data.replace('NR', '0')
    # convert pandas to numpy
    data = np.array(data)

    # delete date and kind
    data = np.delete(data,[0,1,2],1).astype(float) # (4320,24)
    
    splitData = np.transpose(np.array([[0]*18]))

    # concatenate everyday (18,5761)
    for i in range(0, data.shape[0],18):        
        splitData = np.concatenate((splitData,data[i:i+18,:]),axis=1)
    
    # remove first column  (18,5760)
    splitData = np.delete(splitData,[0],1)
    

    # convert -1 to 0.0
    DetectNegData = np.where(splitData[9]<0)
    for i in range(DetectNegData[0].shape[0]):
        splitData[9,DetectNegData[0][i]] = 0.0
    
    # process AMB_TEMP
    air = 0
    threshold = 0
    day = 10
    find = np.where(splitData[air] <= threshold)
    for i in range(find[0].shape[0]):
        splitData[air,find[0][i]] = np.mean(splitData[air,find[0][i]-day:find[0][i]+day])  
    
    # process CH4
    air = 1
    threshold = 1.5
    day = 30
    find = np.where(splitData[air] <= threshold)
    for i in range(find[0].shape[0]):
        splitData[air,find[0][i]] = np.mean(splitData[air,find[0][i]-day:find[0][i]+day]) 
    

    # split to 12 months  (12,18,480)
    monthData = np.array([[[0]*480]*18]*12).astype(float)
    for i in range(12):
        monthData[i] = splitData[:,i*480:(i+1)*480]
    # prepare X, Y data
    X = []
    Y = []
    for i in range(12):
        for j in range(471):
            DetectTooMuchData = np.where(monthData[i,9,j:j+9]>200)
            if (DetectTooMuchData[0].shape[0]>=1 or monthData[i,9,j+9] >= 200):
                continue        
            
            X.append(monthData[i,:,j:j+9])
            Y.append(monthData[i,9,j+9])
    return np.array(X),np.array(Y)


def readTestData(TESTPATH):
    data = pd.read_csv(TESTPATH, sep=",", encoding = "ISO-8859-1" ,header=None)
    data = data.replace('NR', '0')
    data = np.squeeze(np.array(data))
    data = np.delete(data,[0,1],axis=1).astype(float)
    splitData = []
    for i in range(260):
        splitData.append(data[i*18:(i+1)*18,:])
    return np.array(splitData)

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    X = X[randomList,:]
    Y = Y[randomList]
    return X, Y

def train(X, Y, X_val, Y_val ,EPOCH,LEARNINGRATE):
    Y = np.reshape(Y,(-1,1))
    Y_val = np.reshape(Y_val,(-1,1))
    Weight = np.ones((X.shape[1],1))
    Bias = 0.0
    AdagradW = np.zeros((X.shape[1],1))
    AdagradB = 0.0
    lamda = 0.0
    for i in range(EPOCH):
        error = Y - Bias - np.dot(X,Weight)
        errorVal = Y_val - Bias - np.dot(X_val, Weight)
        gradientB = -np.sum(error)
        
        gradientW = -np.dot(np.transpose(X),error) + 2*lamda*Weight

        AdagradW += gradientW ** 2
        AdagradB += gradientB ** 2  
        
        Bias = Bias - LEARNINGRATE / np.sqrt(AdagradB) * gradientB
        Weight = Weight*(1-(LEARNINGRATE / np.sqrt(AdagradW))*lamda) -  LEARNINGRATE / np.sqrt(AdagradW) * gradientW
        
        
        loss = np.sqrt(np.mean(error**2)) + lamda * np.sum(Weight ** 2)
        lossVal = np.sqrt(np.mean(errorVal**2)) + lamda * np.sum(Weight ** 2)
    
        if (i+1) % 1000 == 0:
            print('epoch:{} train loss:{}  validation loss:{}'.format(i+1, loss, lossVal))
    return Weight, Bias, loss
def splitVal(X,Y,size):
    X_rand,Y_rand = shuffle(X,Y)
    X_train, Y_train = X_rand[int(X.shape[0]*size):,:], Y_rand[int(Y.shape[0]*size):]
    X_val, Y_val = X_rand[:int(X.shape[0]*size),:], Y_rand[:int(Y.shape[0]*size)]
    return X_train, Y_train, X_val, Y_val


def predict(X, W, B):
    predictY = np.dot(X,W)+B
    return predictY


def normalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis=0) + 1e-20
    return (X-mean)/std, mean, std   


def main(args):
    # read data
    test = readTestData(args[1])
    
    # select attributes
    selectAttr = ["AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","PM2.5","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]
    attrIndex = []
    for i in range(len(selectAttr)):
        attrIndex.append(ATTR.index(selectAttr[i]))
    test = test[:,attrIndex,:]
    
    
    # reshape to 2D
    test = np.reshape(test,(test.shape[0],-1))  
    
     
    # concatenate CO^1.5 PM10^2 PM2.5^2 WD Direct
    attr = np.array([[2,1.5],[8,2],[9,2],[15,2]])
    for i in attr:
	    test = np.concatenate((test,test[:,int(i[0]*9):int(i[0]*9)+9]**i[1]), axis=1)
    

    parameter = np.loadtxt("parameter.txt")
    mean = parameter[:198]
    std = parameter[198:]

    # normalize
    test = (test-mean)/std
    
    # set parameter
    EPOCH = 1000000
    LEARNINGRATE = 1
    
    # train model and get weight, bias
    # Weight, Bias, Loss = train(X_train, Y_train, X_val, Y_val ,EPOCH, LEARNINGRATE)
    preTrain = np.loadtxt("model.txt")
    Weight = preTrain[:-1].reshape((-1,1))
    Bias = preTrain[-1]


    # predict new data
    predictY = predict(test, Weight, Bias)

    with open(args[2], 'w') as f:
        print('id,value', file=f)
        for (i, p) in enumerate(predictY):
            print('id_{},{}'.format(i, p[0]), file=f)


if __name__ == '__main__':
	main(sys.argv)
