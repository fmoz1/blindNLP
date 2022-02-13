import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Functional API
# i = input(shape = (D,))
# x = Dense(128, activation = 'relu')(i)
# x  = Dense(K, activation = 'softmax')(x)
# model = Model(i, x)
# ...
# model.fit(...)
# model.predict(...)
# this style is good: one letter variable name is used as a convention
# easy to create branches
# easy to define models with multiple i/outputs
# model = Model(inputs = [i1, i2, i3], outputs = [o1, o2, o3])

# Conv1D
# A time-varying signal: Conv1D
# A video: H x W x T: Conv3D
# Voxel: H x W x D: Conv3D (e.g., medical imaging data)

# Conv2D Arguments
# Conv2D(32, (3,3), strides = 2, activation = 'relu', padding = 'same')
# arguments: # output feature maps, filter, speed of filter, activation function, paddding
# Dropout regularization?? (may not be a good idea for CNN.

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print('x_train shape:', x_train.shape)

# the data is 2D, conv expects h x w x c
x_train = np.expand_dim(x_train, -1)  # superfluous dim
x_test = np.expand_dim(x_test, -1)
print('x_train shape:', x_train.shape)

# number of classes
K = len(set(y_train))
print('number of classes:', K)

# build model using the functional api
i = Input(shape=x_train[9].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)  # a feature vector
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)
model = Model(i, x)
model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print('Training model...')
r = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=15
)
# plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
