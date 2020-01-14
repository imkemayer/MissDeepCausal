import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from estimators import tau_dr, tau_ols, tau_ols_ps, get_ps_y01_hat, tau_residuals
from softimpute import get_U_softimpute

#from joblib import Memory
#memory = Memory('cache_dir', verbose=0)


#@memory.cache

def exp_complete(z, x, w, y, **kwargs):
    
    if x is None:
        algo_ = [z] 
        algo_name = ['Z']
    else:   
        algo_ = [z, x]
        algo_name = ['Z', 'X']

    tau = dict()
    for name, zhat in zip(algo_name, algo_):
        
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
        res_tau_ols = tau_ols(zhat, w, y)
        res_tau_ols_ps = tau_ols_ps(zhat, w, y)
        res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat)
        lr = LinearRegression()
        lr.fit(zhat, y)
        y_hat = lr.predict(zhat)
        res_tau_resid = tau_residuals(y, w, y_hat, ps_hat)
        
        tau[name] = res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid

    return tau

def exp_mean(xmiss, w, y, **kwargs):
    from sklearn.impute import SimpleImputer
    x_imp_mean = SimpleImputer(strategy='mean').fit_transform(xmiss)

    ps_hat, y0_hat, y1_hat = get_ps_y01_hat(x_imp_mean, w, y)
    res_tau_ols = tau_ols(x_imp_mean, w, y)
    res_tau_ols_ps = tau_ols_ps(x_imp_mean, w, y)
    res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat)
    lr = LinearRegression()
    lr.fit(x_imp_mean, y)
    y_hat = lr.predict(x_imp_mean)
    res_tau_resid = tau_residuals(y, w, y_hat, ps_hat)

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid


def exp_mf(xmiss, w, y, list_rank = None, **kwargs):
    zhat, r = get_U_softimpute(xmiss, list_rank = list_rank)
    ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
    res_tau_ols = tau_ols(zhat, w, y)
    res_tau_ols_ps = tau_ols_ps(zhat, w, y)
    res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat)
    lr = LinearRegression()
    lr.fit(zhat, y)
    y_hat = lr.predict(zhat)
    res_tau_resid = tau_residuals(y, w, y_hat, ps_hat)

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid, r


def exp_mi(xmiss, w, y, m = 10, **kwargs):

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    res_tau_dr = []
    res_tau_ols = []
    res_tau_ols_ps = []
    res_tau_resid = []
    for i in range(m):
        imp = IterativeImputer(sample_posterior=True, random_state = i)
        ximp_mice = imp.fit_transform(xmiss)
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(ximp_mice, w, y)
        res_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat))
        res_tau_ols.append(tau_ols(ximp_mice, w, y))
        res_tau_ols_ps.append(tau_ols_ps(ximp_mice, w, y))
        lr = LinearRegression()
        lr.fit(ximp_mice, y)
        y_hat = lr.predict(ximp_mice)
        res_tau_resid.append(tau_residuals(y, w, y_hat, ps_hat))

    return np.mean(res_tau_dr), np.mean(res_tau_ols), np.mean(res_tau_ols_ps), np.mean(res_tau_resid)
    


def exp_mdc(xmiss, w, y,
            range_d_miwae=[3, 10],
            range_sig_prior=[0.1, 1, 10], 
            range_num_samples_zmul = [50, 200, 500],
            range_learning_rate = [0.00001, 0.0001, 0.001],
            range_n_epochs=[10, 100, 200], 
             **kwargs):

    from miwae import miwae

    tau = dict()
    tau['MDC.process'] = []
    tau['MDC.mi'] = []

    params = []

    for d_miwae in range_d_miwae:
        for sig_prior in range_sig_prior:
            for num_samples_zmul in range_num_samples_zmul:
                for learning_rate in range_learning_rate:
                    for n_epochs in range_n_epochs:
                        xhat, zhat, zhat_mul = miwae(xmiss, d=d_miwae, sig_prior = sig_prior, 
                                                     num_samples_zmul = num_samples_zmul,
                                                     l_rate = learning_rate, n_epochs=n_epochs)

                        
                        # Tau estimated on Zhat=E[Z|X]
                        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
                        res_tau_ols = tau_ols(zhat, w, y)
                        res_tau_ols_ps = tau_ols_ps(zhat, w, y)
                        res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat)
                        lr = LinearRegression()
                        lr.fit(zhat, y)
                        y_hat = lr.predict(zhat)
                        res_tau_resid = tau_residuals(y, w, y_hat, ps_hat)

                        tau['MDC.process'].append([res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid])

                        # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
                        res_mul_tau_dr = []
                        res_mul_tau_ols = []
                        res_mul_tau_ols_ps = []
                        res_mul_tau_resid = []
                        for zhat_b in zhat_mul: 
                            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat_b, w, y)
                            res_mul_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat))
                            res_mul_tau_ols.append(tau_ols(zhat_b, w, y))
                            res_mul_tau_ols_ps.append(tau_ols_ps(zhat_b, w, y))
                            lr = LinearRegression()
                            lr.fit(zhat_b, y)
                            y_hat = lr.predict(zhat_b)
                            res_mul_tau_resid.append(tau_residuals(y, w, y_hat, ps_hat))

                        res_mul_tau_dr = np.mean(res_mul_tau_dr)
                        res_mul_tau_ols = np.mean(res_mul_tau_ols)
                        res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)
                        res_mul_tau_resid = np.mean(res_mul_tau_resid)

                        tau['MDC.mi'].append([np.mean(res_mul_tau_dr), np.mean(res_mul_tau_ols), np.mean(res_mul_tau_ols_ps), np.mean(res_mul_tau_resid)])

                        params.append([d_miwae, sig_prior, num_samples_zmul, learning_rate, n_epochs])

    return tau, params


if __name__ == '__main__':

    from config import args
    args['range_n_epochs'] = [3]
    args['model'] = "lrmf"
    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, citcio = False, prop_miss = 0, seed = 0)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, citcio = citcio, prop_miss = 0, seed = 0)
    X_miss = ampute(X, prop_miss = 0.1, seed = 0)

    print('test exp with default arguments on miwae')
    tau = exp_mdc(xmiss=X_miss, w=w, y=y, **args)
    print('Everything went well.')