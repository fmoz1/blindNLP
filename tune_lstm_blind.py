# tune neural network model with smac and tpe
# using blind text data
# basics
from ast import Global
import itertools
import os
from unittest import result
from sympy import hyper
import tensorflow as tf
import random as python_random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn and keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, GlobalMaxPooling1D, MaxPooling1D, Conv1D
from keras.layers import Embedding, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# text processing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# tuning: scikit-optimize and hyperopt
from skopt import forest_minimize, gp_minimize, gbrt_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from hyperopt import fmin, tpe, space_eval, Trials
from hyperopt import hp


# for reproduceable result
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)


# set current directory
os.chdir('/Users/fangfeishu/Projects/advancedNLP')

# import custom metrics
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

# model params
MAX_SEQUENCE_LENGTH = 400  # truncate posts at length 400
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 2048  # make sure each batch has a chance of obtaining positive examples
EPOCHS = 5  # change epochs

# load data
file_path = './datafiles/blindPosts/'  # original csvs separated by company
files = [file_path + f for f in os.listdir(file_path)]
df = pd.concat([pd.read_csv(f, parse_dates=True).iloc[:, 1:]
                for f in files])  # concat all csvs
# prepare data
df = df.rename({'post_like': 'likes', 'post_comment': 'comments'}, axis=1)
df['post_firm'] = df['post_firm'].map(lambda x: str(
    x).replace('/company/', '').rstrip('/'))  # op's employer
df['likes'] = df['likes'].map(lambda x: str(x).replace(',', ''))  # of likes
df['comments'] = df['comments'].map(
    lambda x: str(x).replace(',', ''))  # of comments

# change data types
df['likes'] = pd.to_numeric(df['likes'])
df['comments'] = pd.to_numeric(df['comments'])
df['post_timestamp'] = pd.to_datetime(df['post_timestamp'])
df = df.reset_index().set_index(
    ['company', 'post_timestamp']).sort_index().reset_index()

# finalize sample, give labels
df[~df['post_text'].duplicated()]
df['popular'] = np.where(df['likes'] > df['likes'].quantile(0.99), 1, 0)
df['controversial'] = np.where(
    df['comments'] > df['comments'].quantile(0.99), 1, 0)
# keep poster company if available
df = df[['post_text', 'post_firm', 'popular', 'controversial']]

# imbalanced data correction
# examine imbalance
neg, pos = np.bincount(df['popular'])
total = neg + pos
INITIAL_BIAS = np.log([pos/neg])
# class weights
# scaling by total/2 helps keep the loss to a similar magnitude.
# the sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
CLASS_WEIGHT = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('./datafiles/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

# prepare text samples and their labels
print('Loading in posts')
train = df  # we use the entire data
sentences = train['post_text'].fillna('DUMMY_VALUE').values
possible_labels = ['controversial', 'popular']

targets = train[possible_labels].values  # N * 2 matrix

s = sorted(len(s) for s in sentences)
print("median sequence length: ", s[len(s)//2])
print("max sequence length: ", s[-1])
print("min sequence length: ", s[0])
# convert the sentences (strings) to ints
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# get word -> int mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))
# pad sequences so that we get N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor: ', data.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(data, targets, stratify=targets,
                                                    test_size=0.2, random_state=42)
print('Training set: ', X_train.shape, y_train.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False,  # keep the embeddings fixed
)


def create_model(embedding_dim, learning_rate, num_hidden_units, activation):
    print('Filling pre-trained embeddings...')
    num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word2idx.items():
        if i < MAX_VOCAB_SIZE:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        num_words,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False,
    )
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = Bidirectional(LSTM(num_hidden_units, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(len(possible_labels), activation=activation)(x)
    model = Model(input_, output)
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def create_rnn(
        learning_rate,
        num_hidden_units,
        activation):
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = Bidirectional(LSTM(num_hidden_units, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(len(possible_labels), activation=activation)(x)
    model = Model(input_, output)
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=METRICS)
    return model


def create_cnn(learning_rate, num_conv_layers, num_dense_layers, num_dense_nodes, activation):

    model = Sequential()
    # conv layers
    for i in range(num_conv_layers):
        model.add(Conv1D(filters=128, kernel_size=3, activation=activation))
        model.add(MaxPooling1D(pool_size=3))
    # flatten
    model.add(GlobalMaxPooling1D())
    # dense layers
    for i in range(num_dense_layers):
        model.add(Dense(num_dense_nodes, activation=activation))

    # output
    model.add(Dense(len(possible_labels), activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  )
    return model


dim_learning_rate = Real(
    low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate',
)
dim_num_hidden_units = Integer(low=5, high=64, name='num_hidden_units')

dim_activation = Categorical(
    categories=['relu', 'sigmoid'], name='activation',
)

dim_embedding_dims = Integer(low=100, high=200, name='embedding_dims')

dim_num_conv_layers = Integer(low=1, high=3, name='num_conv_layers')

dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')


# param grid
param_grid = [dim_learning_rate,
              dim_num_hidden_units,
              dim_activation]
# Define the objective

path_best_model = './model/tune_blind.h5'
best_accuracy = 0


@use_named_args(param_grid)
def objective(
    learning_rate,
    num_hidden_units,
    activation,
):
    """
    Hyper-parameters:
    embedding_dims:    Embedding dimensions.
    learning_rate:     Learning-rate for the optimizer.
    num_hidden_units:  Number of hidden units for LSTM.
    activation:        Activation function.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_conv_layers:', int(num_hidden_units))
    print('activation:', activation)
    print()

    # Create the neural network with the hyper-parameters.
    # We call the function we created previously.
    model = create_rnn(learning_rate=learning_rate, num_hidden_units=num_hidden_units,
                       activation=activation)

    # model = create_model(embedding_dims=embedding_dims, learning_rate=learning_rate, num_hidden_units=num_hidden_units,
    #                     activation=activation)

    # Set a learning rate annealer
    # this reduces the learning rate if learning does not improve
    # for a certain number of epochs
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',  #
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # train the model
    # we use 3 epochs to be able to run the notebook in a "reasonable"
    # time. If we increase the epochs, we will have better performance
    # this could be another parameter to optimize in fact.
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        class_weight=CLASS_WEIGHT,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=learning_rate_reduction)

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_auc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        # Training CNNs is costly, so we want to avoid having to re-train
        # the network with the best found parameters. We save it instead
        # as we search for the best hyperparam space.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Remember that Scikit-optimize always minimizes the objective
    # function, so we need to negate the accuracy (because we want
    # the maximum accuracy)
    return -accuracy


# test run
default_params = [1e-2, 15, 'relu']
# objective(x=default_params)

# tuning with smac
fm_ = forest_minimize(
    objective,
    param_grid,
    x0=default_params,
    acq_func='EI',
    n_calls=30,
    random_state=0,
    verbose=1,
)

# gp_minimize performs by default GP Optimization
# using a Marten Kernel

""" gp_ = gp_minimize(
    objective,  # the objective function to minimize
    param_grid,  # the hyperparameter space
    x0=default_params,  # the initial parameters to test
    acq_func='EI',  # the acquisition function
    n_calls=30,  # the number of subsequent evaluations of f(x)
    random_state=0,
)
 """

# Analyze results
print(f'Best score: {round(fm_.fun,4)}')
print(f'Best params: {fm_.x}')
print(f'Hyperparameter space: {fm_.space}')

# Convergence
plot_convergence(fm_)

# Partial dependency plot
dim_names = ['learning_rate', 'num_hidden_units', 'activation']
plot_objective(result=fm_, plot_dims=dim_names)
plt.show()

# Eval
plot_evaluations(result=fm_, plot_dims=dim_names)
plt.show()

# best model
model = load_model(path_best_model)

result = model.evaluate(X_test, y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)
