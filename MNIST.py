#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)


# In[4]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255


# In[5]:


train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


# In[6]:


model = Sequential()


# In[7]:


model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[8]:


model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


model.add(Flatten())
model.add(Dense(64))


# In[10]:


model.add(Dense(10))
model.add(Activation('softmax'))


# In[11]:


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[12]:


model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=5)


# In[13]:


test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)


# In[14]:


predictions = model.predict(test_X)
print(np.argmax(np.round(predictions[0])))


# In[15]:


plt.imshow(test_X[0].reshape(28, 28), cmap = plt.cm.binary)
plt.show()

