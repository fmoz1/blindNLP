# optimize a cnn with sciki-optimize with random forests (SMAC)
import itertools
import os
import tensorflow as tf
import random as python_random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
# from keras.optimizers import Adam
import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from skopt import forest_minimize, gbrt_minimize  # different from gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

# for reproduceable result
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)


# set current directory
os.chdir('/Users/fangfeishu/Projects/advancedNLP')


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
X, y = get_mnist()

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,  # the target
    test_size=0.1,
    random_state=0

)

# reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)  # last 1 refers to color bw

# target encoding: one-hot encoding
print(f'Number of classes: {len(set(y_train))}')  # 0-9 10 classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(y_train)  # new target

# print some examples
g = plt.imshow(X_train[1][:, :, 0])
print(f'True Target: {np.argmax(y_train[1])}')


def create_cnn(
    learning_rate,
    num_conv_layers,  # new hp: # of layers
    num_dense_layers,
    num_dense_nodes,
    activation,
):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Start construction of a Keras Sequential model.
    model = Sequential()
    # first layer
    for i in range(num_conv_layers):
        model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                         activation=activation))
    model.add(MaxPool2D(pool_size=2, strides=2))
    # second layer
    for i in range(num_conv_layers):
        model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                         activation=activation))
    model.add(MaxPool2D(pool_size=2, strides=2))
    # Flatten
    model.add(Flatten())
    # Last fully-connected dense layer with softmax-activation
    # for use in classification.
    for i in range(num_dense_layers):
        model.add(Dense(num_dense_nodes, activation=activation))
    model.add(Dense(10, activation='softmax'))

    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)

    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
# Define the hyperparameter space


dim_learning_rate = Real(
    low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate',
)
dim_num_conv_layers = Integer(low=1, high=3, name='num_conv_layers')

dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')


dim_activation = Categorical(
    categories=['relu', 'sigmoid'], name='activation',
)

# param grid
param_grid = [dim_learning_rate,
              dim_num_conv_layers,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]

# Define the objective

path_best_model = './model/cnn_model_fm.h5'
best_accuracy = 0


@use_named_args(param_grid)
def objective(
    learning_rate,
    num_conv_layers,
    num_dense_layers,
    num_dense_nodes,
    activation,
):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_conv_layers:', num_conv_layers)
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()

    # Create the neural network with the hyper-parameters.
    # We call the function we created previously.
    model = create_cnn(learning_rate=learning_rate, num_conv_layers=num_conv_layers,
                       num_dense_layers=num_dense_layers,
                       num_dense_nodes=num_dense_nodes,
                       activation=activation)

    # Set a learning rate annealer
    # this reduces the learning rate if learning does not improve
    # for a certain number of epochs
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
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
                        epochs=3,
                        batch_size=128,
                        validation_split=0.1,
                        callbacks=learning_rate_reduction)

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_accuracy'][-1]

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
default_params = [1e-5, 1, 1, 16, 'relu']
objective(x=default_params)

# Bayesian Opt with RF

fm_ = forest_minimize(
    objective,
    param_grid,
    x0=default_params,
    acq_func='EI',
    n_calls=30,
    random_state=0
)

# Analyze results
print(f'Best score: {round(fm_.fun,4)}')
print(f'Best params: {fm_.x}')
print(f'Hyperparameter space: {fm_.space}')

# Convergence
plot_convergence(fm_)

# Partial dependency plot
dim_names = ['learning_rate', 'num_conv_layers',
             'num_dense_layers', 'num_dense_nodes', 'activation']
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
    # loss 0.0323 accuracy 0.9886
