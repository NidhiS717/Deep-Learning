
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
import glob
import h5py


# In[ ]:


path_train = 'C:/Users/Desktop/AML/PROJECT/fruits-360_dataset_2018_02_08/fruits-360/Training'      #training dataset path
path_valid_dataset = 'C:/Users/Desktop/AML/PROJECT/fruits-360_dataset_2018_02_08/fruits-360/Validation'  #validation dataset path 
apple='C:/Users/Desktop/AML/PROJECT/apple.jpg'
print(apple)

# In[ ]:


m = 28736                    # Training set size
c = 60                      # Number of classes

input_x = np.ndarray((m,64,64,3))
input_y = np.ndarray((m,1))
li = os.listdir(path_train)
li.sort()
classes = np.transpose(np.array(li))
print(classes)
# In[ ]:


# Prepare a Train Dataset.

i=0
label = 0
for e in li:
    image_files = glob.glob(os.path.join(path_train,e) + '/*.jpg')   
    for x in image_files:        
        img = cv2.imread(x,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        input_x[i] = img
        input_y[i] = label
        i=i+1
    label = label +1


# In[ ]:

img = cv2.imread(apple,cv2.IMREAD_COLOR)
img = cv2.resize(img, (64, 64))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
apple_x = img
# apple_y = label
        


# Prepare a Validation dataset

valid_datsetsize = 9673
valid_x = np.ndarray((valid_datsetsize,64,64,3))
valid_y = np.ndarray((valid_datsetsize,1))
li1 = os.listdir(path_valid_dataset)
li1.sort()

#print ('classes:',li1)
#print ('valid_x.shape:',valid_x.shape)
#print ('valid_y.shape:',valid_y.shape)


# In[ ]:


i=0
label = 0
for e in li1:
    image_files = glob.glob(os.path.join(path_valid_dataset,e) + '/*.jpg')
        
    for x in image_files:
        img = cv2.imread(x,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        valid_x[i] = img
        valid_y[i] = label
        i=i+1
    label = label +1


# In[ ]:


from sklearn.utils import shuffle
from keras.models import  Sequential
from keras.layers.core import  Dense, Flatten
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam,RMSprop
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

from sklearn import svm


# In[ ]:


input_x = input_x / 255.
valid_x = valid_x / 255.
apple_x = apple_x/255

y_one_hot  = to_categorical(input_y,c)          #One_hot encoding to input_y (Label) & c  = 60 (no of classes)
valid_y_onehot = to_categorical(valid_y,c)

x_train, y_train = shuffle(input_x, y_one_hot)
X_valid , Y_valid = shuffle(valid_x, valid_y_onehot)

#print ('x_train,y_train:', x_train.shape,y_train.shape)
#print ('X_valid,Y_valid:', X_valid.shape,Y_valid.shape)


# In[ ]:

input_shape = (64,64,3)

model = Sequential()

model.add(Conv2D(16,(3,3),activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(60,activation = 'linear'))

"""
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(60,activation = 'softmax'))
"""

model.compile(loss='hinge', optimizer='sgd', metrics = ['accuracy'])

model.fit(x_train,y_train,batch_size = 32,epochs = 10)

"""
score = model.evaluate(X_valid, Y_valid)
print ('valid_set Accuracy:', score[1] * 100)
print ('valid_set Loss:', score[0])
"""
print(Y_valid.shape,X_valid.shape)
features=model.predict(X_valid)
print(features.shape)
clf=svm.SVC(kernel='rbf',gamma=0.2, C=1)

target = []
for i in Y_valid:
	for j in range(len(i)):
		if(i[j]==1):
			target.append(j)
target = np.array(target)
print(target.shape)
clf.fit(features,target)

a=apple_x.reshape((1,64,64,3))

apple_feature=model.predict(a)

print(clf.predict(apple_feature))
"""

model_json = model.to_json()

with open("fruitsmodel.json", "w") as json_file:
    json_file.write(model_json)
    
#Serialize the weights to HDF5
model.save_weights("fruitweights.h5")
print("Successfully saved model to disk")

"""
