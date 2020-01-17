from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import time
import itertools

from mdc import exp_mdc, exp_mi, exp_mf, exp_mean, exp_complete
from generate_data import gen_lrmf, gen_dlvm, ampute

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', 'exp_name_template', 'Experiment name.')
flags.DEFINE_string('output', None, 'Output path.')

flags.DEFINE_enum('model', None, ['dlvm', 'lrmf'],
                  'Data model class, can be `dlvm` or `lrmf`.')
flags.DEFINE_integer('n_observations', None, 'Number of observations.')
flags.DEFINE_integer('p_ambient', None, 'Dimesion of the ambient space.')
flags.DEFINE_float('snr', None, 'SNR in outcome generation (y0, y1).')
flags.DEFINE_float('prop_miss', None, 'Proportion of MCAR missing values.')
flags.DEFINE_bool('regularize', None, 'Regularize ATE.')
flags.DEFINE_integer('n_seeds', 100, 'Number of seed replications.')
flags.DEFINE_float('d_over_p', None, 'Ratio of d over p.')
flags.DEFINE_integer('n_imputations', None, 'Number of imputations.')

flags.DEFINE_integer('miwae_d_offset', None,
                     'proxy of dim. of latent space given by d + offset.')
flags.DEFINE_float('miwae_sig_prior', None,
                   'Variance of prior distribution on Z for MIWAE.')
flags.DEFINE_integer('miwae_n_samples_zmul', None,
                     'Number of samples from posterior Z|X* for MIWAE.')
flags.DEFINE_float('miwae_learning_rate', None, 'MIWAE learning rate.')
flags.DEFINE_integer('miwae_n_epochs', None,
                     'Number of training epochs for MIWAE.')

# Column names
## Method parameters
l_method_params = ['m','r', 'd_miwae', 'sig_prior',
                   'num_samples_zmul', 'learning_rate',
                   'n_epochs', 'elbo']
## ATE estimator names
l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'tau_resid']

def main(unused_argv):

  # Data generating process parameters
  exp_parameter_grid = {
      'model': ["dlvm", "lrmf"] if FLAGS.model is None else [FLAGS.model],
      'citcio': [False, ],
      'n': [1000, 10000, 100000] if FLAGS.n_observations is None else [FLAGS.n_observations],
      'p': [10, 100, 1000] if FLAGS.p_ambient is None else [FLAGS.p_ambient],
      'snr': [1., 5., 10.] if FLAGS.snr is None else [FLAGS.snr],
      'prop_miss': [0, 0.1, 0.3, 0.5, 0.7, 0.9] if FLAGS.prop_miss is None else [FLAGS.prop_miss],
      'regularize': [False, True] if FLAGS.regularize is None else [FLAGS.regularize],
      'seed': np.arange(FLAGS.n_seeds),
  }
  range_d_over_p = [0.002, 0.01, 0.1] if FLAGS.d_over_p is None else [FLAGS.d_over_p]

  # MDC parameters
  range_d_offset = [0, 5, 10] if FLAGS.miwae_d_offset is None else [FLAGS.miwae_d_offset]

  mdc_parameter_grid = {
      'sig_prior': [0.1, 1, 10] if FLAGS.miwae_sig_prior is None else [FLAGS.miwae_sig_prior],
      'num_samples_zmul': [50, 500] if FLAGS.miwae_n_samples_zmul is None else [FLAGS.miwae_n_samples_zmul],
      'learning_rate': [0.0001,] if FLAGS.miwae_learning_rate is None else [FLAGS.miwae_learning_rate],
      'n_epochs': [500,] if FLAGS.miwae_n_epochs is None else [FLAGS.miwae_n_epochs],
  }

  # MI parameters
  range_m = [10, 20, 50] if FLAGS.n_imputations is None else [FLAGS.n_imputations]

  # Experiment and output file name
  output = f'results/{FLAGS.exp_name}.csv' if FLAGS.output is None else FLAGS.output

  logging.info('*'*20)
  logging.info(f'Starting exp: {FLAGS.exp_name}')
  logging.info('*'*20)
  l_scores = []

  exp_arguments = [dict(zip(exp_parameter_grid.keys(), vals))
                   for vals in itertools.product(*exp_parameter_grid.values())]

  for args in exp_arguments:
    # For given p, create range for d such that 1 < d < p
    # starting with given ratios for d/p
    range_d = [np.maximum(2, int(np.floor(args['p']*x))) for x in range_d_over_p]
    range_d = np.unique(np.array(range_d)[np.array(range_d)<args['p']].tolist())
    for args['d'] in range_d:
      logging.info(f'exp with {args}')

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
      tau = exp_complete(Z, X, w, y, args['regularize'])
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
      tau = exp_mean(X_miss, w, y, args['regularize'])
      args['time'] = int(time.time() - t0)
      l_scores.append(np.concatenate((['Mean_imp'],
                                      list(args.values()),
                                      [None]*len(l_method_params),
                                      tau)))

      # Multiple imputation
      tau = []
      t0 = time.time()
      for m in range_m:
          tau.append(exp_mi(X_miss, w, y, regularize=args['regularize'], m=m))
      args['time'] = int(time.time() - t0)
      for i in range(len(tau)):
          l_scores.append(np.concatenate((['MI'],
                                          list(args.values()),
                                          [m],
                                          [None]*(len(l_method_params)-1),
                                          tau[i])))

      # Matrix Factorization
      t0 = time.time()
      tau = exp_mf(X_miss, w, y, args['regularize'])
      args['time'] = int(time.time() - t0)
      l_scores.append(np.concatenate((['MF'],
                                      list(args.values()),
                                      [None],
                                      [tau[-1]],
                                      [None]*(len(l_method_params)-2),
                                      tau[:-1])))


      # MissDeepCausal
      mdc_parameter_grid['d_miwae'] = [args['d']+x for x in range_d_offset]
      t0 = time.time()

      mdc_arguments = [dict(zip(mdc_parameter_grid.keys(), vals))
                       for vals in itertools.product(*mdc_parameter_grid.values())]

      for mdc_arg in mdc_arguments:
        tau, params = exp_mdc(X_miss, w, y,
                              d_miwae=mdc_arg['d_miwae'],
                              sig_prior=mdc_arg['sig_prior'],
                              num_samples_zmul=mdc_arg['num_samples_zmul'],
                              learning_rate=mdc_arg['learning_rate'],
                              n_epochs=mdc_arg['n_epochs'],
                              regularize=args['regularize'])
        l_scores.append(np.concatenate((['MDC.process'],
                                        list(args.values()),
                                        [None]*2,
                                        params,
                                        tau['MDC.process'])))
        l_scores.append(np.concatenate((['MDC.mi'],
                                        list(args.values()),
                                        [None]*2,
                                        params,
                                        tau['MDC.mi'])))

      args['time'] = int(time.time() - t0)


      score_data = pd.DataFrame(l_scores,
                                columns=['Method'] + list(args.keys()) + l_method_params + l_tau)
      score_data.to_csv(output + '_temp')
      logging.info('........... DONE')
      logging.info(f'in {args["time"]} s \n\n')

  logging.info(f'saving {FLAGS.exp_name} at: {output}')
  score_data.to_csv(output)

  logging.info('*'*20)
  logging.info(f'Exp: {FLAGS.exp_name} succesfully ended.')
  logging.info('*'*20)


if __name__ == "__main__":
  app.run(main)
