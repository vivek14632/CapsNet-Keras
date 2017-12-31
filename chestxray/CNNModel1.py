# Created by Raushan : Start

import os
import sys
import inspect
import datetime
import numpy as np
#import matplotlib.pyplot as plt 
from datetime import datetime
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras import backend as K

# For Calculating total Time taken, while running the model
initialTime = datetime.now()
# Path to images Folder
path_to_image= os.path.dirname(os.path.abspath(__file__)) + "/images/train/"

files=os.listdir(path_to_image) 

path_to_csv=os.path.dirname(os.path.abspath(__file__)) + "/Data_Entry_2017.csv"

fp=open(path_to_csv)
datam=fp.readlines()[1:]
RESHAPED = 1048576
datam_new=[]
XMatrix_test = []
count=0
VERBOSE=1
for row in datam:
	row=row.split(',')
	temp=[]
	temp.append(row[0])
	if(row[1]=="Pneumonia"):
		temp.append(1)
	else:
		temp.append(0)
	datam_new.append(temp)

XMatrix=[]
XMatrix_test=[]
YMatrix_test=[]
count =0

for file in files:
	image = misc.imread(path_to_image+file)
	if (sum(image.shape) == 2048):
			
			#image1=image.reshape(1024,1024,4)
			XMatrix.append(image)
			count+=1
#XMatrix = XMatrix.reshape()
XMatrix=np.asarray(XMatrix)
XMatrix= XMatrix.reshape(count, RESHAPED)
testPercentage = 40
counter = int((testPercentage*count)/100)
TestCounter = counter

for x in range(TestCounter):#np.nditer(XMatrix,flags=['external_loop'], order='F'):
	if(counter>0):
		XMatrix_test.append(XMatrix[x])
		counter-=1
	else:
		break
XMatrix_test=np.asarray(XMatrix_test)
XMatrix_test= XMatrix_test.reshape(TestCounter, RESHAPED)		

		
YMatrix=[]

for file in files:
	for row in datam_new:
		if(file==row[0]):
			YMatrix.append(row[1])

YMatrix=np.asarray(YMatrix)
counter = int((testPercentage*count)/100)
for y in range(TestCounter):#np.nditer(YMatrix, order='F'):
	if(counter>0):
		YMatrix_test.append(YMatrix[y])
		counter-=1
	else:
		break
YMatrix_test=np.asarray(YMatrix_test)
#YMatrix_test= YMatrix_test.reshape(TestCounter, RESHAPED)


#YMatrix= YMatrix.reshape(49, RESHAPED)
input_shape = (1024,1024,1)
model = Sequential()
#model.add(Dense(10,input_shape=(RESHAPED,)))
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
#model.add(Convolution2D(16, 5, 5, activation ='relu', input_shape=input_shape))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(32, (4, 4)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, (4, 4)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
model.add(Dense(64))

model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(XMatrix, YMatrix, batch_size=10, epochs=5,verbose=VERBOSE)
#model.predict(XMatrix_test, batch_size=10, verbose=VERBOSE)
score=model.evaluate(XMatrix_test, YMatrix_test, verbose=VERBOSE)
print("Score is ====> ",score )
finalTime = datetime.now()
totalTime = finalTime - initialTime
print("Total Time the model run is : ",totalTime)
# Created by Raushan : End
