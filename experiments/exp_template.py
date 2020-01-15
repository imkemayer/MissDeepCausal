import numpy as np
import pandas as pd
import time

from main import exp_mdc, exp_mi, exp_mf, exp_mean, exp_complete
from generate_data import gen_lrmf, gen_dlvm, ampute
from config import args

# Data generating process parameters 
range_model = ["dlvm","lrmf"]
range_citcio = [False, True]
range_n = [1000, 5000, 10000, 20000]
range_p = [10, 50, 100, 200]
range_d_over_p = [0.3, 0.6, 0.9]
range_sd = [0.1, 1, 5]
range_prop_miss = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
range_seed = np.arange(100) # to replicate 100 times each experiment

# MDC parameters
range_sig_prior = [0.1, 1, 10]
range_num_samples_zmul = [50, 200, 500]
range_learning_rate = [0.00001, 0.0001, 0.001]
range_n_epochs = [10, 100, 200, 500]

# MI parameters
range_m = [10, 20, 50]

# Output file name
exp_name = 'exp_name0'
 

print('starting exp: ' + exp_name)
l_method_params = ['m','r', 'd_miwae', 'sig_prior', 'num_samples_zmul', 'learning_rate', 'n_epochs']
l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'tau_resid']
output = 'results/'+exp_name+'.csv'
l_scores = []

for args['model'] in range_model:
    for args['citcio'] in range_citcio:
        for args['n'] in range_n:
            for args['p'] in range_p:
                range_d = [np.floor(args['p']*x) for x in range_d_over_p]
                for args['d'] in range_d:
                    for args['sd'] in range_sd:
                        for args['prop_miss'] in range_prop_miss:
                            for args['seed'] in range_seed:
                                if args['model'] == "lrmf":
                                    Z, X, w, y, ps = gen_lrmf(n=args['n'], d=args['d'], p=args['p'], sd = args['sd'],
                                                              citcio = args['citcio'], prop_miss = args['prop_miss'], 
                                                              seed = args['seed'])
                                elif args['model'] == "dlvm":
                                    Z, X, w, y, ps = gen_dlvm(n=args['n'], d=args['d'], p=args['p'], sd = args['sd'],
                                                              citcio = args['citcio'], prop_miss = args['prop_miss'], 
                                                              seed = args['seed'])
                                
                                X_miss = ampute(X, prop_miss = args['prop_miss'], seed = args['seed'])

                                # Complete
                                t0 = time.time()
                                tau = exp_complete(Z, X, w, y)
                                args['time'] = int(time.time() - t0)
                                l_scores.append(np.concatenate((['Z'], list(args.values()), [None]*7, tau['Z'])))
                                l_scores.append(np.concatenate((['X'], list(args.values()), [None]*7, tau['X'])))
                                

                                # Mean-imputation
                                t0 = time.time()
                                tau = exp_mean(X_miss, w, y)
                                args['time'] = int(time.time() - t0)
                                l_scores.append(np.concatenate((['Mean_imp'], list(args.values()), [None]*7, tau)))

                                # MI
                                tau = []
                                t0 = time.time()
                                for m in range_m:
                                    tau.append(exp_mi(X_miss, w, y, m=m))
                                args['time'] = int(time.time() - t0)
                                for i in range(len(tau)):
                                    l_scores.append(np.concatenate((['MI'], list(args.values()), [m], [None]*6, tau[i])))

                                # MF
                                t0 = time.time()
                                tau = exp_mf(X_miss, w, y)
                                args['time'] = int(time.time() - t0)
                                l_scores.append(np.concatenate((['MF'], list(args.values()), [None], [tau[-1]], [None]*5, tau[:-1])))


                                # MDC
                                range_d_miwae = [args['d']+x for x in [0, 5, 10]]
                                t0 = time.time()
                                tau, params = exp_mdc(X_miss, w, y, 
                                                      range_d_miwae = range_d_miwae,
                                                      range_sig_prior = range_sig_prior, 
                                                      range_num_samples_zmul = range_num_samples_zmul,
                                                      range_learning_rate = range_learning_rate,
                                                      range_n_epochs = range_n_epochs)
                                args['time'] = int(time.time() - t0)
                                for i in range(len(tau['MDC.process'])):    
                                    l_scores.append(np.concatenate((['MDC.process'], list(args.values()), [None]*2, params[i], tau['MDC.process'][i])))
                                for i in range(len(tau['MDC.mi'])):    
                                    l_scores.append(np.concatenate((['MDC.mi'], list(args.values()), [None]*2, params[i], tau['MDC.mi'][i])))
                                    
                            print('exp with ', args)
                            print('........... DONE')
                            print('in ', int(args["time"]) , ' s  \n\n')

                            score_data = pd.DataFrame(l_scores, columns=['Method'] + list(args.keys()) + l_method_params + l_tau)
                            score_data.to_csv(output + '_temp')

print('saving ' +exp_name + 'at: ' + output)
score_data.to_csv(output)

print('*'*20)
print('Exp: '+ exp_name+' succesfully ended.')
print('*'*20)