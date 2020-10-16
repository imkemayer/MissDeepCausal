from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import time
import itertools
import csv

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
#tf.logging.set_verbosity(tf.logging.ERROR) # to suppress warning and info messages


from json import dumps
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

#import sys
#sys.path.insert(1,'..')

import os.path
import glob
import pickle

from miwae import miwae_es
from generate_data import gen_lrmf, gen_dlvm, ampute
from utils import compute_rv, mmd
from softimpute_cv import cv_softimpute, softimpute



FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', 'exp_name_template', 'Experiment name.')
flags.DEFINE_string('output', None, 'Output path.')
flags.DEFINE_string('log_path', None, 'Filepath to save the execution state.')

flags.DEFINE_enum('model', None, ['dlvm', 'lrmf'],
                  'Data model class, can be `dlvm` or `lrmf`.')
flags.DEFINE_integer('n_observations', None, 'Number of observations.')
flags.DEFINE_integer('p_ambient', None, 'Dimension of the ambient space.')
flags.DEFINE_float('y_snr', None, 'SNR in outcome generation (y0, y1).')
flags.DEFINE_float('x_snr', None, 'SNR in covariate generation (X).')
flags.DEFINE_float('prop_miss', None, 'Proportion of MCAR missing values.')
flags.DEFINE_bool('regularize', None, 'Regularize ATE.')
flags.DEFINE_integer('n_seeds', 100, 'Number of seed replications.')
flags.DEFINE_float('d_over_p', None, 'Ratio of d over p.')
flags.DEFINE_multi_integer('d_latent', None, 'Dimension of latent space (specify either `d_over_p` or `d`).')
flags.DEFINE_float('mu_z', None, 'Expectation of distribution on Z.')
flags.DEFINE_float('sig_z', None, 'Variance of distribution on Z.')
flags.DEFINE_enum('sig_xgivenz', None, ['fixed', 'random'],'Fixed or random variance for X|Z=z, can be `fixed` or `random`')

flags.DEFINE_integer('miwae_d_offset', None,
                     'proxy of dim. of latent space given by d + offset.')
flags.DEFINE_float('miwae_mu_prior', None,
                   'Expectation of prior distribution on Z for MIWAE.')
flags.DEFINE_float('miwae_sig_prior', None,
                   'Variance of prior distribution on Z for MIWAE.')
flags.DEFINE_integer('miwae_n_samples_zmul', None,
                     'Number of samples from posterior Z|X* for MIWAE.')
flags.DEFINE_float('miwae_learning_rate', None, 'MIWAE learning rate.')
flags.DEFINE_integer('miwae_n_epochs', None,
                     'Number of training epochs for MIWAE.')


def log_res(path, results, keys):
  write_header = False
  if not tf.io.gfile.exists(path):
    write_header=True

  with tf.io.gfile.GFile(path, 'a') as f:
    csv_writer = csv.DictWriter(f, fieldnames=keys)
    if write_header:
      csv_writer.writeheader()
    for res in results:
      csv_writer.writerow(res)


def main(unused_argv):
  # Data generating process parameters
  exp_parameter_grid = {
     'model': ["dlvm", "lrmf"] if FLAGS.model is None else [FLAGS.model],
     'citcio': [False, ],
     'n': [1000, 10000, 100000] if FLAGS.n_observations is None else [FLAGS.n_observations],
     'p': [10, 100, 1000] if FLAGS.p_ambient is None else [FLAGS.p_ambient],
     'y_snr': [5.] if FLAGS.y_snr is None else [FLAGS.y_snr],
     'x_snr': 1.*np.arange(2,20,4) if FLAGS.x_snr is None else [FLAGS.x_snr],
     'mu_z': [0.] if FLAGS.mu_z is None else [FLAGS.mu_z],
     'sig_z': [1.] if FLAGS.sig_z is None else [FLAGS.sig_z],
     'sig_xgivenz': ["fixed", ] if FLAGS.sig_xgivenz is None else [FLAGS.sig_xgivenz],
     'prop_miss': [0.0, 0.1, 0.3, 0.5] if FLAGS.prop_miss is None else [FLAGS.prop_miss],
     'regularize': [False] if FLAGS.regularize is None else [FLAGS.regularize],
     'seed': np.arange(FLAGS.n_seeds),
  }
  range_d_over_p = [0.002, 0.01, 0.1] if FLAGS.d_over_p is None and FLAGS.d_latent is None else [FLAGS.d_over_p]
  range_d = None if range_d_over_p is not None and FLAGS.d_latent is None else FLAGS.d_latent

  # MDC parameters
  range_d_offset = [0, 5, 10] if FLAGS.miwae_d_offset is None else [FLAGS.miwae_d_offset]

  mdc_parameter_grid = {
     'mu_prior': [0.] if FLAGS.miwae_mu_prior is None else [FLAGS.miwae_mu_prior],
     'sig_prior': [1.] if FLAGS.miwae_sig_prior is None else [FLAGS.miwae_sig_prior],
     'num_samples_zmul': [500] if FLAGS.miwae_n_samples_zmul is None else [FLAGS.miwae_n_samples_zmul],
     'learning_rate': [0.0001,] if FLAGS.miwae_learning_rate is None else [FLAGS.miwae_learning_rate],
     'n_epochs': [5000,] if FLAGS.miwae_n_epochs is None else [FLAGS.miwae_n_epochs],
  }



  # Experiment and output file name
  output = f'results/{FLAGS.exp_name}.csv' if FLAGS.output is None else FLAGS.output

  FLAGS.log_dir = './sessions/logging/' if FLAGS.log_path is None else FLAGS.log_path
  logging.get_absl_handler().use_absl_log_file()
  logging.info('*'*20)
  logging.info(f'Starting exp: {FLAGS.exp_name}')
  logging.info('*'*20)

  exp_arguments = [dict(zip(exp_parameter_grid.keys(), vals))
                  for vals in itertools.product(*exp_parameter_grid.values())]

  previous_runs = set()
  if tf.io.gfile.exists(output):
    with tf.io.gfile.GFile(output, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
          # Note: we need to do this conversion because DictReader creates an
          # OrderedDict, and reads all values as str instead of bool or int.
          previous_runs.add(str({
             'model': row['model'],
             'citcio': row['citcio'] == 'True',
             'n': int(row['n']),
             'p': int(row['p']),
             'y_snr': float(row['y_snr']),
             'x_snr': float(row['x_snr']),
             'mu_z': float(row['mu_z']),
             'sig_z': float(row['sig_z']),
             'prop_miss': float(row['prop_miss']),
             'regularize': row['regularize'] == 'True',
             'seed': int(row['seed']),
             'd': int(row['d']),
             'sig_xgivenz': row['sig_xgivenz']
          }))
  logging.info('Previous runs')
  logging.info(previous_runs)

  for args in exp_arguments:
    # For given p, if range_d is not yet specified,
    # create range for d such that 1 < d < p
    # starting with given ratios for d/p
    if range_d is None:
        range_d = [np.maximum(2, int(np.floor(args['p']*x))) for x in range_d_over_p]
        range_d = np.unique(np.array(range_d)[np.array(range_d)<args['p']].tolist())

    exp_time = time.time()
    for args['d'] in range_d:
      # We only consider cases where latent dimension <= ambient dimension
      if args['d'] > args['p']:
          continue
      res = []

      if str(args) in previous_runs:
        logging.info(f'Skipped {args}')
        continue
      else:
        logging.info(f'running exp with {args}')

      if args['model'] == "lrmf":
        Z, X, w, y, ps, mu0, mu1 = gen_lrmf(n=args['n'], d=args['d'], p=args['p'],
                                            y_snr=args['y_snr'], x_snr=args['x_snr'],
                                            citcio=args['citcio'],
                                            prop_miss=args['prop_miss'],
                                            seed=args['seed'])
      elif args['model'] == "dlvm":
        Z, X, w, y, ps, mu0, mu1 = gen_dlvm(n=args['n'], d=args['d'], p=args['p'],
                                            y_snr=args['y_snr'], citcio=args['citcio'],
                                            prop_miss=args['prop_miss'],
                                            seed=args['seed'],
                                            mu_z=args['mu_z'],
                                            sig_z=args['sig_z'],
                                            x_snr=args['x_snr'],
                                            sig_xgivenz=args['sig_xgivenz'])

      X_miss = ampute(X, prop_miss = args['prop_miss'], seed = args['seed'])


      # MIWAE
      mdc_parameter_grid['d_miwae'] = [args['d']+x for x in range_d_offset]

      mdc_arguments = [dict(zip(mdc_parameter_grid.keys(), vals))
                       for vals in itertools.product(*mdc_parameter_grid.values())]

      for mdc_arg in mdc_arguments:
          t0 = time.time()
          mdc_arg['mu_prior']=args['mu_z']
          session_file = './sessions/' + \
                              args['model'] + '_'+ \
                              args['sig_xgivenz'] + 'Sigma'+ \
                              '_n' + str(args['n']) + \
                              '_p' + str(args['p']) + \
                              '_d' + str(args['d']) + \
                              '_ysnr' + str(args['y_snr']) +\
                              '_xsnr' + str(args['x_snr']) +\
                              '_propNA' + str(args['prop_miss']) + \
                              '_seed' + str(args['seed'])
          session_file_complete = session_file + \
                                  '_dmiwae' + str(mdc_arg['d_miwae']) + \
                                  '_sigprior' + str(mdc_arg['sig_prior'])
          epochs=-1
          tmp = glob.glob(session_file_complete+'.*')
          sess = tf.Session(graph=tf.reset_default_graph())
          if len(tmp)>0:
            continue
          else:
            xhat, zhat, zhat_mul, elbo, epochs = miwae_es(X_miss,
                                                          d_miwae=mdc_arg['d_miwae'],
                                                          mu_prior=mdc_arg['mu_prior'],
                                                          sig_prior=mdc_arg['sig_prior'],
                                                          num_samples_zmul=mdc_arg['num_samples_zmul'],
                                                          l_rate=mdc_arg['learning_rate'],
                                                          n_epochs=mdc_arg['n_epochs'],
                                                          save_session = True,
                                                          session_file = session_file)
            with open(session_file_complete + '.pkl', 'wb') as file_data:  # Python 3: open(..., 'wb')
              pickle.dump([xhat, zhat, zhat_mul, elbo, epochs], file_data)

          exp_time = int(time.time() - t0)


      logging.info('........... DONE')
      logging.info(f'in {time.time() - exp_time} s \n\n')

  logging.info('*'*20)
  logging.info(f'Exp: {FLAGS.exp_name} successfully ended.')
  logging.info('*'*20)

if __name__ == "__main__":
  app.run(main)
