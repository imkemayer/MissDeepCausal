import numpy as np
from math import sqrt
from sklearn.preprocessing import scale


def compute_rv(X,Y, standardize=False):
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
    if standardize:
        n = X.shape[0]
        beta_x = np.square(np.trace(W1))/np.trace(np.square(W1))
        beta_y = np.square(np.trace(W2))/np.trace(np.square(W2))
        mean_h0 = np.sqrt(betax*betay)/(n-1.)
        tau_x = (n-1.)/((n-3.)*(n-1-beta_x)) * \
                    (n*(n+1)*np.sum(np.diag(np.square(W1)))/np.trace(np.square(W1)) - \
                    (n-1)*(beta_x+2))
        tau_y = (n-1.)/((n-3.)*(n-1-beta_y)) * \
                    (n*(n+1)*np.sum(np.diag(np.square(W2)))/np.trace(np.square(W2)) - \
                    (n-1)*(beta_y+2))
        var_h0 = 2*(n-1-beta_x)*(n-2-beta_y)/((n+1)*(n-1)**2*(n-2)) * \
                    (1+(n-3)/(2*n*(n-1)) * tau_x*tau_y)
        rv = rv-mean_h0/np.sqrt(var_h0)
    return rv


# Code taken from Werner Zellinger
# 2020-09-23, https://github.com/wzell/mann/blob/master/models/maximum_mean_discrepancy.py
def mmd(x1, x2, beta):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
    return diff

def gaussian_kernel(x1, x2, beta = 1.0):
    #r = x1.dimshuffle(0,'x',1)
    r = x1
    return np.exp( -beta * np.square(r - x2).sum(axis=-1))

def add_identity(axes, label=None, *line_args, **line_kwargs):
    identity, = axes.plot([], [], label=label, *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes
