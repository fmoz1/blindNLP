# use logistic regression for mnist classification assignment
# without scikit-learn
# only with basic libraries (numpy, matplotlib and pandas )
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# set current directory
os.chdir('/Users/fangfeishu/Projects/advancedNLP')


def get_mnist():
    if not os.path.exists('./datafiles/digit-recognizer'):
        print('No mnist folder in the directory')
    if not os.path.exists('./datafiles/digit-recognizer/train.csv'):
        print('No train data.')
    df = pd.read_csv('./datafiles/digit-recognizer/train.csv')
    data = df.to_numpy().astype(np.float32)
    np.random.shuffle(data)  # shuffle the data
    X = data[:, 1:]
    Y = data[:, 0]

    # train test split
    X_train = X[:-1000]
    y_train = Y[:-1000]
    X_test = X[-1000:]
    y_test = Y[-1000:]

    # normalize the data
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # are they all zero?
    idx = np.where(std == 0)[0]
    assert(np.all(std[idx] == 0))  # 0 divisor

    np.place(std, std == 0, 1)  # replace 0 divisor by 1
    X_train = (X_train - mu)/std
    X_test = (X_test - mu)/std
    return X_train, X_test, y_train, y_test


def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.show()
    return P


def forward(X, W, b):
    # softmax
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()


def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)


def predict(p_y):
    return np.argmax(p_y, axis=1)


def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    K = y.max() + 1
    ind = np.zeros((N, K))
    for n in range(N):
        k = y[n]
        ind[n, k] = 1
    return ind


def gradW(t, y, X):
    return X.T.dot(t-y)


def gradb(t, y):
    return (t-y).sum(axis=0)  # no regularization


def linear_benchmark():
    X_train, X_test, y_train, y_test = get_mnist()
    print('Peforming logistic regression')

    # convert y_train and y_test to (N x K) matrices of indicators
    N, D = X_train.shape
    y_train_ind = y2indicator(y_train)
   # print(y_train_ind)
    y_test_ind = y2indicator(y_test)
    K = y_train_ind.shape[1]

    W = np.random.randn(D, K)/np.sqrt(D)
    b = np.zeros(K)
    train_losses = []
    test_losses = []
    train_classification_errors = []
    test_classification_errors = []

    lr = 0.00003  # learning rate
    reg = 0.0
    n_iters = 100
    for i in range(n_iters):
        p_y = forward(X_train, W, b)
        train_loss = cost(p_y, y_train_ind)
        train_losses.append(train_loss)
        train_err = error_rate(p_y, y_train)
        train_classification_errors.append(train_err)

        p_y_test = forward(X_test, W, b)
        test_loss = cost(p_y_test, y_test_ind)
        test_losses.append(test_loss)
        test_err = error_rate(p_y_test, y_test)
        test_classification_errors.append(test_err)

        # gradient descent
        W += lr*(gradW(y_train_ind, p_y, X_train) - reg*W)
        b += lr*(gradb(y_train_ind, p_y))

        if (i+1) % 10 == 0:
            print(f'Iter: {i+1}/{n_iters}, Train loss: {train_loss:.3f}'
                  f'Train error: {train_err:.3f}, Test loss: {test_loss:.3f}'
                  f'Test error: {test_err:.3f}')

    p_y = forward(X_test, W, b)
    print('Final error rate: ', error_rate(p_y, y_test))

    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Loss per iteration')
    plt.legend()
    plt.show()

    plt.plot(train_classification_errors, label='Train err')
    plt.plot(test_classification_errors, label='Test err')
    plt.title('Classification errors per iteration')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    linear_benchmark()
