import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation



ds = pd.read_csv('mnistTest.csv')
# print(ds.head())
data = np.array(ds)
x_data = data[:,1:]
y_data = data[:,0]
x_data=x_data/255.0

Y=np_utils.to_categorical(y_data)
x_train = x_data[:8000,:]
x_val = x_data[8000:,:]
y_train = Y[:8000]
y_val = Y[8000:]
# print(x_train[2])
# print(y_train[2])

model = Sequential()

model.add(Dense(256,input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=256,epochs=50,verbose=2,validation_data=(x_val,y_val))

plt.plot(history.history['loss'],'g',label='loss')
plt.plot(history.history['val_loss'],'b',label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'],'b',label='accuracy')
plt.plot(history.history['val_acc'],'r',label='val_accuracy')
plt.legend()
plt.show()
