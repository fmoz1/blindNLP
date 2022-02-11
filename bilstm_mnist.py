import os
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, \
    Lambda, Concatenate, Dense  # lambda, concat are new imports
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.3


def get_mnist(limit=None):
    if not os.path.exists('./datafiles/digit-recognizer'):
        print('No mnist folder in the directory')
    if not os.path.exists('./datafiles/digit-recognizer/train.csv'):
        print('No train data.')
    df = pd.read_csv('./datafiles/digit-recognizer/train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:].reshape(-1, 28, 28) / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


# get data
X, Y = get_mnist()

# config
D = 28  # input dim (28 x 28)
M = 15  # hidden dim

#  input is an image of size 28x28
input_ = Input(shape=(D, D))

# up-down
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_)  # output N x D x 2M
x1 = GlobalMaxPooling1D()(x1)  # N x 2M

# left right
rnn2 = Bidirectional(LSTM(M, return_sequences=True))
permutor = Lambda(lambda t: K.permute_dimensions(
    t, pattern=(0, 2, 1)))  # permute_dimensions()
x2 = permutor(input_)
x2 = rnn2(input_)  # output N x D x 2M
x2 = GlobalMaxPooling1D()(x2)  # N x 2M

# concat
concatenator = Concatenate(axis=1)
x = concatenator([x1, x2])

# final dense layer
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_, outputs=output)


model.compile(
    loss='sparse_categorical_crossentropy',  # no need for one-hot
    optimizer='adam',
    metrics=['accuracy']
)

print('Training model...')
r = model.fit(
    X,
    Y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
)
# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()


model.save('./model/bilstm_mnist_model')
