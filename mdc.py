import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from estimators import compute_estimates
from generate_data import gen_lrmf, gen_dlvm, ampute
from miwae import miwae_es
from softimpute_cv import softimpute, cv_softimpute


def exp_complete(z, x, w, y):
    algo_ = [z, x]
    algo_name = ['Z', 'X']

    tau = dict()
    for name, zhat in zip(algo_name, algo_):
        tau_tmp = compute_estimates(zhat, w, y)
        tau[name] = list(tau_tmp.values())

    return tau

def exp_mean(xmiss, w, y):
    x_imp_mean = SimpleImputer(strategy = 'mean').fit_transform(xmiss)

    tau_tmp = compute_estimates(x_imp_mean, w, y)

    return list(tau_tmp.values())


def exp_mf(xmiss, w, y, grid_len=15):
    err, grid = cv_softimpute(xmiss, grid_len = grid_len)
    zhat, _ = softimpute(xmiss, lamb = grid[np.argmin(err)])
    tau_tmp = compute_estimates(zhat, w, y)

    return np.concatenate((list(tau_tmp.values()), [zhat.shape[1]]))


def exp_mi(xmiss, w, y, m=10):

    res_tau_dr = []
    res_tau_ols = []
    res_tau_ols_ps = []
    res_tau_resid = []
    for i in range(m):
        imp = IterativeImputer(sample_posterior = True, random_state = i)
        x_imp_mice = imp.fit_transform(xmiss)
        tau_tmp = compute_estimates(x_imp_mice, w, y)
        res_tau_dr.append(tau_tmp['tau_dr'])
        res_tau_ols.append(tau_tmp['tau_ols'])
        res_tau_ols_ps.append(tau_tmp['tau_ols_ps'])
        res_tau_resid.append(tau_tmp['tau_resid'])

    return np.mean(res_tau_dr), np.mean(res_tau_ols), np.mean(res_tau_ols_ps), np.mean(res_tau_resid)



def exp_mdc(xmiss, w, y,
            d_miwae,
            sig_prior,
            num_samples_zmul,
            learning_rate,
            n_epochs):

    tau = dict()

    xhat, zhat, zhat_mul, elbo = miwae_es(xmiss, d=d_miwae, sig_prior=sig_prior,
                                          num_samples_zmul=num_samples_zmul,
                                          l_rate=learning_rate, n_epochs=n_epochs)
    # Tau estimated on Zhat=E[Z|X]
    tau_tmp = compute_estimates(zhat, w, y)
    tau['MDC.process'] = list(tau_tmp.values())

    # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
    res_mul_tau_dr = []
    res_mul_tau_ols = []
    res_mul_tau_ols_ps = []
    res_mul_tau_resid = []
    for zhat_b in zhat_mul:
        tau_tmp = compute_estimates(zhat_b, w, y)
        res_mul_tau_dr.append(tau_tmp['tau_dr'])
        res_mul_tau_ols.append(tau_tmp['tau_ols'])
        res_mul_tau_ols_ps.append(tau_tmp['tau_ols_ps'])
        res_mul_tau_resid.append(tau_tmp['tau_resid'])
    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)
    res_mul_tau_resid = np.mean(res_mul_tau_resid)
    tau['MDC.mi'] = [np.mean(res_mul_tau_dr), np.mean(res_mul_tau_ols),
                     np.mean(res_mul_tau_ols_ps), np.mean(res_mul_tau_resid)]

    params = [d_miwae, sig_prior, num_samples_zmul, learning_rate, n_epochs, elbo]

    return tau, params
