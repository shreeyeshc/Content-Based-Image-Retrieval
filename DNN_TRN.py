import numpy as np
from scipy.io import wavfile
#import StringIO
import cv2
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
from scipy.stats import skew,kurtosis
import pickle
import numpy


# DNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum
 
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)


def load_wav(filename,samplerate=44100):
    # load file
    rate, data = wavfile.read(filename)
    # convert stereo to mono
    if len(data.shape) > 1:
        data = data[:,0]/2 + data[:,1]/2

    # re-interpolate samplerate
    ratio = float(samplerate) / float(rate)
    #data = resample(data, len(data) * ratio)
    return samplerate, data.astype(np.int16)

def DNN_TRAIN():
    # Train model
    print ('Training')
    mypath='samples/'
    Img = []
    labels=[];
    FV=[];
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    images = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        I=cv2.imread(join(mypath,onlyfiles[n]))
        #2:PREPROCESS
        # RESIZING
        I= cv2.resize(I,(32,32),0)
        #3:ENHANCEMENT
        # MEDIAN FILTER
        I = cv2.medianBlur(I,5)
        # FEATURE EXTRACTION
        # DWT
        I=I[:,:,0]
        coeffs2 = pywt.dwt2(I, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

        FV1=np.median(LL) 
        FV2=np.mean(LL)
        FV3=skew(skew(LL))
        FV4=kurtosis(kurtosis(LL))
        
        FV5=np.median(LH) 
        FV6=np.mean(LH)
        FV7=skew(skew(LH))
        FV8=kurtosis(kurtosis(LH))
        
        FV9=np.median(HL) 
        FV10=np.mean(HL)
        FV11=skew(skew(HL))
        FV12=kurtosis(kurtosis(HL))
        
        FV13=np.median(HH) 
        FV14=np.mean(HH)
        FV15=skew(skew(HH))
        FV16=kurtosis(kurtosis(HH))
        
        TST=np.array([FV1,FV2,FV3,FV4,FV5,FV6,FV7,FV8,FV9,FV10,FV11,FV12,FV13,FV14,FV15,FV16])
        print(TST)
        print(type(TST))
        Img.append(TST)
        labels.append(n+1)

    print(np.shape(Img))
    H=np.array(Img)
    print(type(H))
    FV=[]
    for i in range(600):
        dataset=H[i,:]
        # split into train and test sets
        train_size = int(len(dataset) * 0.5)
        test_size = len(dataset) - train_size
        print('TRAIN SIZE',train_size)
        print('TEST SIZE',test_size)
        train = dataset[0:train_size];
        test=dataset[train_size:len(dataset)]
        print('TRAIN SIZE',train)
        print('TEST SIZE',test)


        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


        # TRAIN 
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)


        # TEST 
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        testPredict=np.transpose(testPredict);
        testPredict=np.mean(testPredict)
        print('FV:',np.shape(testPredict))
        FV.append(testPredict)

    print(np.shape(FV))
    np.savetxt('FV.txt',FV)

def Calc_Wt2(TST):
        file= open("CNN_CBIR.obj",'rb')
        TRR = pickle.load(file)
        file.close()
        print(type(TRR))
        WTRN = np.array(TRR)
        R, C = np.shape(WTRN)
        print("R",R)
        print("C",C)
        M = []
        ERR = []
        WTST = TST
        R, C = np.shape(WTRN)
        for i in range(0, R):
            RR = WTRN[i,:]
            Temp = np.subtract(WTST, RR)
            ERR = LA.norm(Temp)
            M.append(ERR)
        ind = np.argmin(M);
        print('ERROR',np.min(M))
        #  RETRIVAL
        CLASSD=[]
        HH=np.loadtxt('NEWLABEL.txt')
        for item in HH:
            CLASSD.append(np.round(item))

        CLASSD=np.asarray(CLASSD)
        #CLASSD = np.asarray(HH)
        IND=CLASSD[ind]
        print('INDEX IN TOTAL DATASET',ind)
        print('CLASS',IND)
        CLASS=np.where(CLASSD==IND)
        CLASSD=[]
        for item in CLASS:
            CLASSD.append(np.round(item))
        CLASS=np.asarray(CLASSD)
        CLASS=np.transpose(CLASS)
        #print("CLASS:",CLASS)
        print('LENGTH',len(CLASS))
        NTRN=[]
        LBLS=[]
        for i in range(0, len(CLASS)):
            mm=CLASS[i]
            RR = WTRN[mm,:]
            RR=np.array(RR)
            RR=np.transpose(RR)
            print(np.shape(RR))
            NTRN.append(RR)
            LBLS.append(mm)

        NTRN=np.array(NTRN)
        print('SORTED CLASS',(CLASS))
        R, C,m = np.shape(NTRN)
        ERRS=M
        M=[]
        LBL=[]
        for i in range(0, R):
            print(CLASS[i])
            ERR = ERRS[int(CLASS[i])]
            LBL.append(int(LBLS[i]))
            M.append(ERR)
        INC=np.min(LBL)
        ind = INC+np.argmin(M);
        print('ind',np.argsort(M))
        ind2=INC+np.argsort(M)
        print(ind2)
        ind2=ind2[0:5]
        return ind,ind2
    