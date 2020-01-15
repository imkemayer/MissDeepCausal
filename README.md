# MissDeepCausal

Code for MissDeepCausal paper.

Structure:
```
.
|  generate_data.py
|    Main functions
|    - gen_lrmf: generate data (Z,X,W,Y) according to LRMF model
|    - gen_dlvm: generate data (Z,X,W,Y) according to DLVM model
|    Helper functions
|    - get_dlvm_params: generate parameters of conditional (normal) distribution of X | Z
|    - citcio_treat_out: generate (W,Y) under "unconfoundedness despite missingness"
|    - gen_treat: generate W as a function of confounders (with default link = "linear")
|    - gen_outcome: generate Y as a function of confounders and W 
|                   (with default link = "linear")
|    - ampute: generate missing values under MCAR mechanism
|
|  miwae.py
|    - miwae: MIWAE as proposed by P.-A. Mattei with additional sampling from Z|X 
|             using self-normalized importance sampling weights
|    - miwae_cv: cross-validation for sigma (variance of prior on Z) 
|                and for d_miwae (dimension of latent space)
|
|  estimators.py
|    Main functions
|    - compute_estimates: Computes the estimators below on given data
|    - tau_dr: ATE estimation via parametric AIPW (using output of get_ps_y01_hat)
|    - tau_ols: ATE estimation via regression of Y on W and (covariates or confounders) 
|    - tau_ols_ps: ATE estimation via regression of Y on W and (covariates or confounders) and PS
|                  (using PS estimation from get_ps_y01_hat)
|    - tau_residuals: ATE estimation via residuals on residuals regression (using output of get_ps_y01_hat)
|    Helper functions
|    - get_ps_y01_hat: estimate propensity score and regression functions using logistic and linear regression
|    
|
|  main.py
|    - exp_complete: ATE estimation on complete data (X or Z) using estimators from estimators.py
|    - exp_mean: ATE estimation on mean-imputed data using estimators from estimators.py
|    - exp_mf: ATE estimation on approximate latent factors obtained with SoftImpute using estimators from estimators.py
|    - exp_mi: ATE estimation via multiple imputation using estimators from estimators.py
|    - exp_mdc: ATE estimation via MDC (process and mi) using estimators from estimators.py
|
|  config.py: default values for data generation
|
|  softimpute.py: WORK IN PROGRESS (SoftImpute implementation from iskandr/fancyimpute)
|
|  Rsoftimpute.py: WORK IN PROGRESS (import of R function SoftImpute)



./experiments
|  exp_template.py: template for calling all functions of main.py on data generated with different sets of parameters 
|                   (for data generation, for mdc and for mi)

./results
|  *.csv: Experiment results in form of csv files (containing output of exp_template.py) are saved here
|
```

Note that treatment assignment vector is assumed to have values in {0, 1} (not {-1, 1}).


To use R package `softImpute`, use the rpy2 module and install the R package via:
```
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate()
base = importr('base')
utils = importr('utils')

packnames = ('lattice', 'Matrix', 'softImpute')
from rpy2.robjects.vectors import StrVector
utils.chooseCRANmirror(ind=1)
utils.install_packages(StrVector(packnames))
```
