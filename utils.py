import numpy as np
from math import sqrt
from sklearn.preprocessing import scale


def compute_rv(X,Y):
    assert X.shape[0]==Y.shape[0], 'Need the same dimension for X and Y'
    if X.shape[0] == 1:
        return np.NaN
    if len(X.shape)==1:
        X=X.reshape([X.shape[0],1])
    if len(Y.shape)==1:
        Y=Y.reshape([Y.shape[0],1])
    X = scale(X, with_std=False)
    Y = scale(Y, with_std=False)
    W1 = np.dot(X, np.transpose(X))
    W2 = np.dot(Y, np.transpose(Y))
    rv = np.trace(np.dot(W1, W2))/sqrt(np.trace(np.dot(W1, W1))* np.trace(np.dot(W2, W2)))
    return rv