import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import os.path
import glob
import pickle


from estimators import compute_estimates
from generate_data import gen_lrmf, gen_dlvm, ampute
from miwae import miwae_es
from softimpute_cv import softimpute, cv_softimpute


def exp_complete(z, x, w, y, regularize, nuisance=False):
    algo_ = [z, x]
    algo_name = ['Z', 'X']

    tau = dict()
    nu = dict()
    for name, zhat in zip(algo_name, algo_):
        if nuisance:
            tau_tmp, nu_tmp = compute_estimates(zhat, w, y, regularize, nuisance)
            nu[name] = nu_tmp
        else:
            tau_tmp = compute_estimates(zhat, w, y, regularize)
        tau[name] = tau_tmp

    if nuisance:
        return tau, nu
    return tau

def exp_mean(xmiss, w, y, regularize, nuisance=False):
    x_imp_mean = SimpleImputer(strategy = 'mean').fit_transform(xmiss)

    return compute_estimates(x_imp_mean, w, y, regularize, nuisance)


def exp_mf(xmiss, w, y, regularize, nuisance=False, return_zhat=False, grid_len=15):
    err, grid = cv_softimpute(xmiss, grid_len = grid_len)
    zhat, _ = softimpute(xmiss, lamb = grid[np.argmin(err)])

    if nuisance:
        tau,nu = compute_estimates(zhat, w, y, regularize, nuisance)
        if return_zhat:
            return tau, nu, zhat.shape[1], zhat
        return tau, nu, zhat.shape[1]
    if return_zhat:
        return compute_estimates(zhat, w, y, regularize), zhat.shape[1], zhat
    return compute_estimates(zhat, w, y, regularize), zhat.shape[1]


def exp_mi(xmiss, w, y, regularize, m=10, nuisance=False):

    res_tau_dr = []
    res_tau_ols = []
    res_tau_ols_ps = []
    res_tau_resid = []
    res_ps = np.empty([len(w),1])
    res_y0 = np.empty([len(y),1])
    res_y1 = np.empty([len(y),1])
    for i in range(m):
        imp = IterativeImputer(sample_posterior = True, random_state = i)
        x_imp_mice = imp.fit_transform(xmiss)
        if nuisance:
            tau_tmp, nu_tmp = compute_estimates(x_imp_mice, w, y, regularize, nuisance)
            res_ps = np.concatenate((res_ps, nu_tmp['ps_hat'].reshape([len(w),1])), axis=1)
            res_y0 = np.concatenate((res_y0, nu_tmp['y0_hat'].reshape([len(y),1])), axis=1)
            res_y1 = np.concatenate((res_y1, nu_tmp['y1_hat'].reshape([len(y),1])), axis=1)
        else:
            tau_tmp = compute_estimates(x_imp_mice, w, y, regularize)
        res_tau_dr.append(tau_tmp['tau_dr'])
        res_tau_ols.append(tau_tmp['tau_ols'])
        res_tau_ols_ps.append(tau_tmp['tau_ols_ps'])
        res_tau_resid.append(tau_tmp['tau_resid'])

    if nuisance:
        return {
            'tau_dr': np.mean(res_tau_dr),
            'tau_ols': np.mean(res_tau_ols),
            'tau_ols_ps': np.mean(res_tau_ols_ps),
            'tau_resid': np.mean(res_tau_resid),
        }, {
            'ps_hat': np.mean(res_ps[:,1:], axis=1),
            'y0_hat': np.mean(res_y0[:,1:], axis=1),
            'y1_hat': np.mean(res_y1[:,1:], axis=1),}
    return {
        'tau_dr': np.mean(res_tau_dr),
        'tau_ols': np.mean(res_tau_ols),
        'tau_ols_ps': np.mean(res_tau_ols_ps),
        'tau_resid': np.mean(res_tau_resid),
    }


def exp_mdc(xmiss, w, y,
            d_miwae,
            mu_prior,
            sig_prior,
            num_samples_zmul,
            learning_rate,
            n_epochs,
            regularize,
            nuisance=False,
            return_zhat=False,
            save_session=False,
            session_file=None,
            session_file_complete=None):

    tau = dict()
    nu = dict()
    epochs=-1
    xhat = []
    zhat = []
    zhat_mul = []
    elbo = None
    epochs = None
    if session_file_complete is not None:
        tmp = glob.glob(session_file_complete+'.*')
        sess = tf.Session(graph=tf.reset_default_graph())
        if len(tmp)>0:
            new_saver = tf.train.import_meta_graph(session_file_complete + '.meta')
            new_saver.restore(sess, session_file_complete)
            with open(session_file_complete+'.pkl', 'rb') as f:
                xhat, zhat, zhat_mul, elbo, epochs = pickle.load(f)
    if session_file_complete is None or len(tmp)==0:
        xhat, zhat, zhat_mul, elbo, epochs = miwae_es(xmiss,
                                                      d_miwae=d_miwae,
                                                      mu_prior=mu_prior,
                                                      sig_prior=sig_prior,
                                                      num_samples_zmul=num_samples_zmul,
                                                      l_rate=learning_rate,
                                                      n_epochs=n_epochs,
                                                      save_session=save_session,
                                                      session_file=session_file)
        if session_file_complete is not None:
            new_saver = tf.train.import_meta_graph(session_file_complete + '.meta')
            new_saver.restore(sess, session_file_complete)#tf.train.latest_checkpoint('./'))
            with open(session_file_complete + '.pkl', 'wb') as file_data:  # Python 3: open(..., 'wb')
                pickle.dump([xhat, zhat, zhat_mul, elbo, epochs], file_data)


    # Tau estimated on Zhat=E[Z|X]
    if nuisance:
        tau['MDC.process'], nu['MDC.process'] = compute_estimates(zhat, w, y, regularize, nuisance)
    else:
        tau['MDC.process'] = compute_estimates(zhat, w, y, regularize)

    # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
    res_mul_tau_dr = []
    res_mul_tau_ols = []
    res_mul_tau_ols_ps = []
    res_mul_tau_resid = []
    res_mul_ps = np.empty([len(w),1])
    res_mul_y0 = np.empty([len(y),1])
    res_mul_y1 = np.empty([len(y),1])
    for zhat_b in zhat_mul:
        if nuisance:
            tau_tmp, nu_tmp = compute_estimates(zhat_b, w, y, regularize, nuisance)
            res_mul_ps = np.concatenate((res_mul_ps, nu_tmp['ps_hat'].reshape([len(w),1])), axis=1)
            res_mul_y0 = np.concatenate((res_mul_y0, nu_tmp['y0_hat'].reshape([len(y),1])), axis=1)
            res_mul_y1 = np.concatenate((res_mul_y1, nu_tmp['y1_hat'].reshape([len(y),1])), axis=1)
        else:
            tau_tmp = compute_estimates(zhat_b, w, y, regularize)
        res_mul_tau_dr.append(tau_tmp['tau_dr'])
        res_mul_tau_ols.append(tau_tmp['tau_ols'])
        res_mul_tau_ols_ps.append(tau_tmp['tau_ols_ps'])
        res_mul_tau_resid.append(tau_tmp['tau_resid'])
    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)
    res_mul_tau_resid = np.mean(res_mul_tau_resid)
    if nuisance:
        nu['MDC.mi'] = {
                'ps_hat': np.mean(res_mul_ps[:,1:], axis=1),
                'y0_hat': np.mean(res_mul_y0[:,1:], axis=1),
                'y1_hat': np.mean(res_mul_y1[:,1:], axis=1)
        }
    tau['MDC.mi'] = {
        'tau_dr': np.mean(res_mul_tau_dr),
        'tau_ols': np.mean(res_mul_tau_ols),
        'tau_ols_ps': np.mean(res_mul_tau_ols_ps),
        'tau_resid': np.mean(res_mul_tau_resid)
    }
    if nuisance:
        if return_zhat:
            return tau, nu, elbo, zhat, zhat_mul
        return tau, nu, elbo
    if return_zhat:
        return tau, elbo, zhat, zhat_mul
    return tau, elbo
