from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import time
import itertools
import csv

import tensorflow as tf


from sklearn.metrics import mean_squared_error
from mdc import exp_mdc, exp_mi, exp_mf, exp_mean, exp_complete
from generate_data import gen_lrmf, gen_dlvm, ampute

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', 'exp_nuisance_test', 'Experiment name.')
flags.DEFINE_string('output', None, 'Output path.')
flags.DEFINE_string('log_path', None, 'Filepath to save the execution state.')

flags.DEFINE_enum('model', None, ['dlvm', 'lrmf'],
                  'Data model class, can be `dlvm` or `lrmf`.')
flags.DEFINE_integer('n_observations', None, 'Number of observations.')
flags.DEFINE_integer('p_ambient', None, 'Dimesion of the ambient space.')
flags.DEFINE_float('y_snr', None, 'SNR in outcome generation (y0, y1).')
flags.DEFINE_float('x_snr', None, 'SNR in covariate generation (X).')
flags.DEFINE_float('prop_miss', None, 'Proportion of MCAR missing values.')
flags.DEFINE_bool('regularize', None, 'Regularize ATE.')
flags.DEFINE_integer('n_seeds', 5, 'Number of seed replications.')
flags.DEFINE_float('d_over_p', None, 'Ratio of d over p.')
flags.DEFINE_multi_integer('d_latent', None, 'Dimension of latent space (specify either `d_over_p` or `d`).')
flags.DEFINE_float('mu_z', None, 'Expectation of distribution on Z.')
flags.DEFINE_float('sig_z', None, 'Variance of distribution on Z.')
flags.DEFINE_float('sig_xgivenz', None, 'Value of fixed variance for X|Z=z, must be positive')

flags.DEFINE_integer('n_imputations', None, 'Number of imputations.')

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


# Column names
## Method parameters
l_method_params = ['m','r', 'd_miwae', 'mu_prior', 'sig_prior',
                   'num_samples_zmul', 'learning_rate',
                   'n_epochs', 'elbo']
## ATE estimator names
l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'tau_resid']
## Nuisance parameter estimation errors
l_nu = ['ps_hat_mse', 'y0_hat_mse', 'y1_hat_mse']


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
      'nuisance':[True,],
      'n': [1000, 5000, 10000] if FLAGS.n_observations is None else [FLAGS.n_observations],
      'p': [5, 10, 100] if FLAGS.p_ambient is None else [FLAGS.p_ambient],
      'y_snr': [5.] if FLAGS.y_snr is None else [FLAGS.y_snr],
      'x_snr': [2.] if FLAGS.x_snr is None else [FLAGS.x_snr],
      'mu_z': [0.] if FLAGS.mu_z is None else [FLAGS.mu_z],
      'sig_z': [1.] if FLAGS.sig_z is None else [FLAGS.sig_z],
      'sig_xgivenz': [0.001] if FLAGS.sig_xgivenz is None else [FLAGS.sig_xgivenz],
      'prop_miss': [0.0,] if FLAGS.prop_miss is None else [FLAGS.prop_miss],
      'regularize': [False] if FLAGS.regularize is None else [FLAGS.regularize],
      'seed': np.arange(FLAGS.n_seeds),
      }
    range_d_over_p = [0.002, 0.01, 0.1] if FLAGS.d_over_p is None and FLAGS.d_latent is None else [FLAGS.d_over_p]
    range_d = None if range_d_over_p is not None and FLAGS.d_latent is None else FLAGS.d_latent

    # MDC parameters
    range_d_offset = [0, 5] if FLAGS.miwae_d_offset is None else [FLAGS.miwae_d_offset]

    mdc_parameter_grid = {
      'mu_prior': [0.] if FLAGS.miwae_mu_prior is None else [FLAGS.miwae_mu_prior],
      'sig_prior': [1.] if FLAGS.miwae_sig_prior is None else [FLAGS.miwae_sig_prior],
      'num_samples_zmul': [500] if FLAGS.miwae_n_samples_zmul is None else [FLAGS.miwae_n_samples_zmul],
      'learning_rate': [0.0001,] if FLAGS.miwae_learning_rate is None else [FLAGS.miwae_learning_rate],
      'n_epochs': [5000,] if FLAGS.miwae_n_epochs is None else [FLAGS.miwae_n_epochs],
      }

    # MI parameters
    range_m = [10,] if FLAGS.n_imputations is None else [FLAGS.n_imputations]

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
                        'sig_xgivenz': float(row['sig_xgivenz'])
                }))
    logging.info('Previous runs')
    logging.info(previous_runs)

    for args in exp_arguments:
        ## For given p, create range for d such that 1 < d < p
        ## starting with given ratios for d/p
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
                                                  seed=args['seed'],
                                                  sig_xgivenz=args['sig_xgivenz'])
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

            # On complete data
            t0 = time.time()
            if args['nuisance']:
                tau, nu = exp_complete(Z, X, w, y, args['regularize'], args['nuisance'])
            else:
                tau = exp_complete(Z, X, w, y, args['regularize'], args['nuisance'])
            args['time'] = int(time.time() - t0)
            row = {'Method': 'Z'}
            row.update(args)
            row.update(tau['Z'])
            print(tau['Z'])
            if args['nuisance']:
                row.update({'ps_hat_mse': mean_squared_error(ps, nu['Z']['ps_hat'])})
                row.update({'y0_hat_mse': mean_squared_error(mu0, nu['Z']['y0_hat'])})
                row.update({'y1_hat_mse': mean_squared_error(mu1, nu['Z']['y1_hat'])})
            res.append(row)
            row = {'Method': 'X'}
            row.update(args)
            row.update(tau['X'])
            if args['nuisance']:
                row.update({'ps_hat_mse': mean_squared_error(ps, nu['X']['ps_hat'])})
                row.update({'y0_hat_mse': mean_squared_error(mu0, nu['X']['y0_hat'])})
                row.update({'y1_hat_mse': mean_squared_error(mu1, nu['X']['y1_hat'])})
            res.append(row)


            # Mean-imputation
            t0 = time.time()
            if args['nuisance']:
                tau, nu = exp_mean(X_miss, w, y, args['regularize'], args['nuisance'])
            else:
                tau = exp_mean(X_miss, w, y, args['regularize'])
            args['time'] = int(time.time() - t0)
            row = {'Method': 'Mean_imp'}
            row.update(args)
            row.update(tau)
            if args['nuisance']:
                row.update({'ps_hat_mse': mean_squared_error(ps, nu['ps_hat'])})
                row.update({'y0_hat_mse': mean_squared_error(mu0, nu['y0_hat'])})
                row.update({'y1_hat_mse': mean_squared_error(mu1, nu['y1_hat'])})
            res.append(row)

            # Multiple imputation
            for m in range_m:
                t0 = time.time()
                if args['nuisance']:
                    tau, nu = exp_mi(X_miss, w, y, regularize=args['regularize'], m=m, nuisance=args['nuisance'])
                else:
                    tau = exp_mi(X_miss, w, y, regularize=args['regularize'], m=m)
                args['time'] = int(time.time() - t0)
                row = {'Method': 'MI', 'm': m}
                row.update(args)
                row.update(tau)
                if args['nuisance']:
                    row.update({'ps_hat_mse': mean_squared_error(ps, nu['ps_hat'])})
                    row.update({'y0_hat_mse': mean_squared_error(mu0, nu['y0_hat'])})
                    row.update({'y1_hat_mse': mean_squared_error(mu1, nu['y1_hat'])})
                res.append(row)

            # Matrix Factorization
            t0 = time.time()
            if args['nuisance']:
                tau, nu, r, zhat = exp_mf(X_miss, w, y, args['regularize'], args['nuisance'], return_zhat=True)
            else:
                tau, r = exp_mf(X_miss, w, y, args['regularize'])
            args['time'] = int(time.time() - t0)
            row = {'Method': 'MF', 'r': r}
            row.update(args)
            row.update(tau)
            if args['nuisance']:
                row.update({'ps_hat_mse': mean_squared_error(ps, nu['ps_hat'])})
                row.update({'y0_hat_mse': mean_squared_error(mu0, nu['y0_hat'])})
                row.update({'y1_hat_mse': mean_squared_error(mu1, nu['y1_hat'])})
            res.append(row)


            # MissDeepCausal
            mdc_parameter_grid['d_miwae'] = [args['d']+x for x in range_d_offset]

            mdc_arguments = [dict(zip(mdc_parameter_grid.keys(), vals))
                           for vals in itertools.product(*mdc_parameter_grid.values())]

            for mdc_arg in mdc_arguments:
                t0 = time.time()
                mdc_arg['mu_prior']=args['mu_z']
                session_file = './sessions/' + \
                                    args['model'] + '_'+ \
                                    '_sigXgivenZ' + str(args['sig_xgivenz']) + \
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
                if args['nuisance']:
                    tau, nu, elbo, zhat, zhat_mul = exp_mdc(X_miss, w, y,
                                      d_miwae=mdc_arg['d_miwae'],
                                      mu_prior=mdc_arg['mu_prior'],
                                      sig_prior=mdc_arg['sig_prior'],
                                      num_samples_zmul=mdc_arg['num_samples_zmul'],
                                      learning_rate=mdc_arg['learning_rate'],
                                      n_epochs=mdc_arg['n_epochs'],
                                      regularize=args['regularize'],
                                      nuisance=args['nuisance'],
                                      return_zhat = True,
                                      save_session=True,
                                      session_file=session_file,
                                      session_file_complete=session_file_complete)
                else:
                    tau, elbo, zhat, zhat_mul = exp_mdc(X_miss, w, y,
                                      d_miwae=mdc_arg['d_miwae'],
                                      mu_prior=mdc_arg['mu_prior'],
                                      sig_prior=mdc_arg['sig_prior'],
                                      num_samples_zmul=mdc_arg['num_samples_zmul'],
                                      learning_rate=mdc_arg['learning_rate'],
                                      n_epochs=mdc_arg['n_epochs'],
                                      regularize=args['regularize'],
                                      return_zhat = True,
                                      save_session=True,
                                      session_file=session_file,
                                      session_file_complete=session_file_complete)


                args['training_time'] = int(time.time() - t0)
                row = {'Method': 'MDC.process', 'elbo': elbo}
                row.update(args)
                row.update(mdc_arg)
                row.update(tau['MDC.process'])
                if args['nuisance']:
                    row.update({'ps_hat_mse': mean_squared_error(ps, nu['MDC.process']['ps_hat'])})
                    row.update({'y0_hat_mse': mean_squared_error(mu0, nu['MDC.process']['y0_hat'])})
                    row.update({'y1_hat_mse': mean_squared_error(mu1, nu['MDC.process']['y1_hat'])})
                res.append(row)
                row = {'Method': 'MDC.mi', 'elbo': elbo}
                row.update(args)
                row.update(mdc_arg)
                row.update(tau['MDC.mi'])
                if args['nuisance']:
                    row.update({'ps_hat_mse': mean_squared_error(ps, nu['MDC.mi']['ps_hat'])})
                    row.update({'y0_hat_mse': mean_squared_error(mu0, nu['MDC.mi']['y0_hat'])})
                    row.update({'y1_hat_mse': mean_squared_error(mu1, nu['MDC.mi']['y1_hat'])})
                res.append(row)

            log_res(output, res, ['Method'] + list(args.keys()) + l_method_params + l_tau + l_nu)
            logging.info('........... DONE')
            logging.info(f'in {time.time() - exp_time} s \n\n')

    logging.info('*'*20)
    logging.info(f'Exp: {FLAGS.exp_name} succesfully ended.')
    logging.info('*'*20)


if __name__ == "__main__":
    app.run(main)
