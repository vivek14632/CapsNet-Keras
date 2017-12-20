# Created by Raushan : Start

import os
import sys
import inspect
import datetime
import numpy as np
import matplotlib.pyplot as plt 
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
path_to_image= os.path.dirname(os.path.abspath(__file__)) + "\images"

files=os.listdir(path_to_image) 
# Path to CSV File
path_to_csv= os.path.dirname(os.path.abspath(__file__)) + "\Data_Entry_2017.csv"

fp=open(path_to_csv)
datam=fp.readlines()[1:]
datam_new=[]

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
count =0

for file in files:
	image = misc.imread(path_to_image+file)
	if (sum(image.shape) == 2048):
			
			image1=np.expand_dims(image,4)
			XMatrix.append(image1)

XMatrix=np.asarray(XMatrix)
YMatrix=[]

for file in files:
	for row in datam_new:
		if(file==row[0]):
			YMatrix.append(row[1])

YMatrix=np.asarray(YMatrix)
input_shape = (1024,1024,1)
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=input_shape))
#model.add(Convolution2D(16, 5, 5, activation ='relu', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(XMatrix, YMatrix, batch_size=10, epochs=1,verbose=1)
model.predict(XMatrix, batch_size=10, verbose=20)

finalTime = datetime.now()
totalTime = finalTime - initialTime
print("Total Time the model run is : ",totalTime)
# Created by Raushan : End
