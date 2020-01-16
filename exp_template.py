import numpy as np
import pandas as pd
import time

from mdc import exp_mdc, exp_mi, exp_mf, exp_mean, exp_complete
from generate_data import gen_lrmf, gen_dlvm, ampute
from config import args

# Data generating process parameters
parameter_grid = {
  'model': ["dlvm", "lrmf"], # data model class
  'citcio': [False, ], # classical unconfoundedness (on Z) or unconfoundedness despite missingness (on X)
  'n': [1000, 10000, 100000], # number of observations
  'p': [10, 100, 1000], # dimension of ambient space
  'd_over_p': [0.002, 0.01, 0.1], # ratio d over p
  'snr': [1., 5., 10.], # SNR in outcome generation (y0, y1)
  'prop_miss': [0, 0.1, 0.3, 0.5, 0.7, 0.9], # proportion of MCAR missing values
  'seed': np.arange(100), # to replicate 100 times each experiment
}

# MDC parameters
range_d_offset = [0, 5, 10] # proxy of dim. of latent space given by d + offset
range_sig_prior = [0.1, 1, 10] # variance of prior distribution on Z
range_num_samples_zmul = [50, 500] # number of samples from posterior Z | X*
range_learning_rate = [0.0001, ] # learning rate of MIWAE
range_n_epochs = [500, ] # number of epochs of MIWAE (combined with early stopping)

# MI parameters
range_m = [10, 20, 50] # number of imputations

# Experiment and output file name
exp_name = 'exp_name_template'
output = '../results/'+exp_name+'.csv'


# Column names
## Method parameters
l_method_params = ['m','r', 'd_miwae', 'sig_prior',
                   'num_samples_zmul', 'learning_rate',
                   'n_epochs', 'elbo']
## ATE estimator names
l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'tau_resid']

def main():
  print('*'*20)
  print('Starting exp: ' + exp_name)
  print('*'*20)
  l_scores = []

  arguments = [dict(zip(parameter_grid.keys(), vals))
               for vals in itertools.product(*parameter_grid.values())]

  for args in arguments:
    # For given p, create range for d such that 1 < d < p
    # starting with given ratios for d/p
    range_d = [np.maximum(2, np.floor(args['p']*x)) for x in range_d_over_p]
    range_d = np.unique(np.array(range_d)[np.array(range_d)<args['p']].tolist())
    for args['d'] in range_d:
      print('exp with ', args)

      if args['model'] == "lrmf":
          Z, X, w, y, ps = gen_lrmf(n=args['n'], d=args['d'], p=args['p'],
                                    y_snr=args['snr'], citcio=args['citcio'],
                                    prop_miss=args['prop_miss'],
                                    seed=args['seed'])
      elif args['model'] == "dlvm":
          Z, X, w, y, ps = gen_dlvm(n=args['n'], d=args['d'], p=args['p'],
                                    y_snr=args['snr'], citcio=args['citcio'],
                                    prop_miss=args['prop_miss'],
                                    seed=args['seed'])

      X_miss = ampute(X, prop_miss = args['prop_miss'], seed = args['seed'])

      # On complete data
      t0 = time.time()
      tau = exp_complete(Z, X, w, y)
      args['time'] = int(time.time() - t0)
      l_scores.append(np.concatenate((['Z'],
                                      list(args.values()),
                                      [None]*len(l_method_params),
                                      tau['Z'])))
      l_scores.append(np.concatenate((['X'],
                                      list(args.values()),
                                      [None]*len(l_method_params),
                                      tau['X'])))


      # Mean-imputation
      t0 = time.time()
      tau = exp_mean(X_miss, w, y)
      args['time'] = int(time.time() - t0)
      l_scores.append(np.concatenate((['Mean_imp'],
                                      list(args.values()),
                                      [None]*len(l_method_params),
                                      tau)))

      # Multiple imputation
      tau = []
      t0 = time.time()
      for m in range_m:
          tau.append(exp_mi(X_miss, w, y, m=m))
      args['time'] = int(time.time() - t0)
      for i in range(len(tau)):
          l_scores.append(np.concatenate((['MI'],
                                          list(args.values()),
                                          [m],
                                          [None]*(len(l_method_params)-1),
                                          tau[i])))

      # Matrix Factorization
      t0 = time.time()
      tau = exp_mf(X_miss, w, y)
      args['time'] = int(time.time() - t0)
      l_scores.append(np.concatenate((['MF'],
                                      list(args.values()),
                                      [None],
                                      [tau[-1]],
                                      [None]*(len(l_method_params)-2),
                                      tau[:-1])))


      # MissDeepCausal
      range_d_miwae = [args['d']+x for x in range_d_offset]
      t0 = time.time()
      tau, params = exp_mdc(X_miss, w, y,
                            range_d_miwae=range_d_miwae,
                            range_sig_prior=range_sig_prior,
                            range_num_samples_zmul=range_num_samples_zmul,
                            range_learning_rate=range_learning_rate,
                            range_n_epochs=range_n_epochs)
      args['time'] = int(time.time() - t0)
      for i in range(len(tau['MDC.process'])):
          l_scores.append(np.concatenate((['MDC.process'],
                                          list(args.values()),
                                          [None]*2,
                                          params[i],
                                          tau['MDC.process'][i])))
      for i in range(len(tau['MDC.mi'])):
          l_scores.append(np.concatenate((['MDC.mi'],
                                          list(args.values()),
                                          [None]*2,
                                          params[i],
                                          tau['MDC.mi'][i])))


      score_data = pd.DataFrame(l_scores,
                                columns=['Method'] + list(args.keys()) + l_method_params + l_tau)
      score_data.to_csv(output + '_temp')
      print('........... DONE')
      print('in {} s \n\n'.format(int(args["time"])))

  print('saving {} at: {}'.format(exp_name, output))
  score_data.to_csv(output)

  print('*'*20)
  print('Exp: {} succesfully ended.'.format(exp_name))
  print('*'*20)


if __name__ == "__main__":
  main()
