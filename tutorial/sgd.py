from tkinter.tix import Tree
from xml.dom.xmlbuilder import DocumentLS
from logit_mnist import get_mnist, forward, error_rate, cost, gradW, gradb, y2indicator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime


def main():
    X_train, X_test, y_train, y_test = get_mnist()  # load data
    print('Performing logistic regression')

    N, D = X_train.shape
    # convert targets to ind matrices
    y_train_ind = y2indicator(y_train)
    y_test_ind = y2indicator(y_test)

    # 1. Full GD
    W = np.random.randn(D, 10) / np.sqrt(D)
    W0 = W.copy()  # initial weights are the same
    b = np.zeros(10)
    test_losses_full = []
    lr = 0.9  # learning rate
    reg = 0.
    t0 = datetime.now()
    last_dt = 0
    intervals = []
    for i in range(50):
        p_y = forward(X_train, W, b)
        gW = gradW(y_train_ind, p_y, X_train) / N  # divided by N = sample size
        gb = gradb(y_train_ind, p_y) / N

        W += lr*(gW - reg*W)
        b += lr*(gb - reg*b)

        p_y_test = forward(X_test, W, b)
        test_loss = cost(p_y_test, y_test_ind)
        dt = (datetime.now() - t0.total_second())

        # save
        dt2 = dt - last_dt
        last_dt = dt
        intervals.append(dt2)

        test_losses_full.append([dt, test_loss])
        if (i+1) % 10 == 0:
            print('Cost at iteration %d: %.6f' % (i+1, test_loss))

    p_y = forward(X_test, W, b)
    print('Final error rate: ', error_rate(p_y, y_test))
    print('Elapsed time for full GD', datetime.now() - t0)

    # save the max time
    max_dt = dt  # quit once reach max dt
    avg_interval_dt = np.mean(intervals)

    # 1. SGD
    W = np.random.randn(D, 10) / np.sqrt(D)
    W0 = W.copy()  # initial weights are the same
    b = np.zeros(10)
    test_losses_sgd = []
    lr = 0.001  # learning rate
    reg = 0.
    t0 = datetime.now()
    last_dt_calculated_loss = 0
    done = False
    intervals = []
    for i in range(50):
        tmpX, tmpY = shuffle(X_train, y_train_ind)
        for n in range(N):
            x = tmpX[n, :].reshape(1, D)  # convert to 2 dim
            y = tmpY[n:, :].reshape(1, 10)
            p_y = forward(X_train, W, b)
            gW = gradW(y_train_ind, p_y, X_train) / \
                N  # divided by N = sample size
            gb = gradb(y_train_ind, p_y) / N

            W += lr*(gW - reg*W)
            b += lr*(gb - reg*b)

            dt = (datetime.now() - t0).total_seconds()
            dt2 = dt - last_dt_calculated_loss

            if dt2 > avg_interval_dt:
                last_dt_calculated_loss = dt
                p_y_test = forward(X_test, W, b)
                test_los = cost(p_y_test, y_test_ind)
                test_losses_sgd.append*[dt, test_loss]

            # time to quit
            if dt > max_dt:
                done = True
                break
        if done:
            break
        if (i+1) % 10 == 0:
            print('Cost at iteration %d: %.6f' % (i+1, test_loss))
