from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np
import matplotlib.pyplot as plt

T = 8  # seq length
D = 2  # input dim
M = 3  # hidden layer size

X = np.random.randn(1, T, D)  # single sentence word vector


def lstm1():

    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)

    print("o:", o)  # actual output
    print("h:", h)  # hidden state
    print("c:", c)  # cell state


def lstm2():

    input_ = Input(shape=(T, D))
    # vs. return_sequences = False
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)

    print("o:", o)  # actual output
    print("h:", h)  # hidden state
    print("c:", c)  # cell state


def gru1():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)

    print("o:", o)  # actual output
    print("h:", h)  # hidden state


def gru2():
    input_ = Input(shape=(T, D))
    # vs. return_sequences = False
    rnn = GRU(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)

    print("o:", o)  # actual output
    print("h:", h)  # hidden state

def bidi1():
    input_ = Input(shape=(T, D))
    rnn = Bidirectional(LSTM(M, return_state=True))
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h1, c1, h2, c2 = model.predict(X)

    print("o:", o)  # actual output
    print("o.shape", o.shape)
    print("h1:", h1)
    print("c1:", c1)
    print("h2:", h2)
    print("c2:", c2)


def bidi2():
    input_ = Input(shape=(T, D))
    rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=True))
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h1, c1, h2, c2 = model.predict(X)

    print("o:", o)  # actual output
    print("o.shape", o.shape)
    print("h1:", h1)
    print("c1:", c1)
    print("h2:", h2)
    print("c2:", c2)


if __name__ == '__main__':
    print('==========lstm1===========')
    lstm1()
    print('==========lstm2===========')
    lstm2()
    print('==========gru1===========')
    gru1()
    print('==========gru2===========')
    gru2()
    print('==========bidrectional1===========')
    bidi1()
    print('==========bidrectional2===========')
    bidi2()

