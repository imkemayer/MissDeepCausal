import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from estimators import compute_estimates
from softimpute_cv import softimpute, cv_softimpute


def exp_complete(z, x, w, y, **kwargs):
    algo_ = [z, x]
    algo_name = ['Z', 'X']

    tau = dict()
    for name, zhat in zip(algo_name, algo_):
        tau_tmp = compute_estimates(zhat, w, y)
        tau[name] = list(tau_tmp.values())

    return tau

def exp_mean(xmiss, w, y, **kwargs):
    from sklearn.impute import SimpleImputer
    x_imp_mean = SimpleImputer(strategy = 'mean').fit_transform(xmiss)

    tau_tmp = compute_estimates(x_imp_mean, w, y)

    return list(tau_tmp.values())


def exp_mf(xmiss, w, y, grid_len = 15, **kwargs):
    err, grid = cv_softimpute(xmiss, grid_len = grid_len)
    zhat,_ = softimpute(xmiss, lamb = grid[np.argmin(err)])
    tau_tmp = compute_estimates(zhat, w, y)
    
    return np.concatenate((list(tau_tmp.values()), [zhat.shape[1]]))


def exp_mi(xmiss, w, y, m = 10, **kwargs):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

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
            range_d_miwae = [3, 10],
            range_sig_prior = [0.1, 1, 10], 
            range_num_samples_zmul = [50, 200, 500],
            range_learning_rate = [0.00001, 0.0001, 0.001],
            range_n_epochs = [10, 100, 200], 
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
                        xhat, zhat, zhat_mul, elbo = miwae(xmiss, d = d_miwae, sig_prior = sig_prior, 
                                                           num_samples_zmul = num_samples_zmul,
                                                           l_rate = learning_rate, n_epochs = n_epochs)
                        # Tau estimated on Zhat=E[Z|X]
                        tau_tmp = compute_estimates(zhat, w, y)
                        tau['MDC.process'].append(list(tau_tmp.values()))

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
                        tau['MDC.mi'].append([np.mean(res_mul_tau_dr), np.mean(res_mul_tau_ols), np.mean(res_mul_tau_ols_ps), np.mean(res_mul_tau_resid)])

                        params.append([d_miwae, sig_prior, num_samples_zmul, learning_rate, n_epochs, elbo])
    
    return tau, params


if __name__ == '__main__':

    from config import args
    from generate_data import gen_lrmf, gen_dlvm, ampute

    args['model'] = "lrmf"
    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n = n, d = d, p = p, citcio = False, prop_miss = 0, seed = 0)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n = n, d = d, p = p, citcio = False, prop_miss = 0, seed = 0)
    X_miss = ampute(X, prop_miss = 0.1, seed = 0)

    print('test exp with multiple imputation using 10 imputations')
    tau = exp_mi(xmiss = X_miss, w = w, y = y, m = 10, **args)
    print('Everything went well.')