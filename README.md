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
|  softimpute.py: SoftImpute implementation from iskandr/fancyimpute
|
|  miwae.py
|    - miwae: MIWAE as proposed by P.-A. Mattei with additional sampling from Z|X 
|             using self-normalized importance sampling weights
|    - miwae_cv: cross-validation for sigma (variance of prior on Z) 
|                and for d_miwae (dimension of latent space)
|
|  estimators.py
|    - tau_mi: ATE estimation via multiple imputation
|    - tau_mia: ATE estimation via mia.grf (not implemented)
|    - tau_grf: ATE estimation via mean.grf (not implemented)
|    - get_ps_y01_hat: estimate propensity score and regression functions using logistic and linear regression
|    - tau_residuals: ATE estimation via residuals on residuals regression (using output of get_ps_y01_hat)
|    - tau_dr: ATE estimation via parametric AIPW (using output of get_ps_y01_hat)
|    - tau_ols: ATE estimation via regression of Y on W and (covariates or confounders) 
|    - tau_ols_ps: ATE estimation via regression of Y on W and (covariates or confounders) and PS
|                  (using PS estimation from get_ps_y01_hat)
|
|  main.py
|    - exp_complete: ATE estimation on complete data (X or Z) with synthetic data
|    - exp_mi: ATE estimation via multiple imputation with synthetic data
|    - exp_cevae: ATE estimation via CEVAE on mean imputation with synthetic data
|    - exp_miwae: ATE estimation via MIWAE with synthetic data
|
|  main_ihdp.py
|    - ihdp_baseline: ATE estimation on complete data (X or Z) with IHDP data
|    - ihdp_mi: ATE estimation via multiple imputation with IHDP data
|    - ihdp_cevae: ATE estimation via CEVAE on mean imputation with IHDP data
|    - ihdp_miwae_save: save output of MIWAE (E[Z|X] and B samples from P(Z|X)) with IHDP data
|    - ihdp_miwae: ATE estimation via MIWAE with IHDP data
|    - ihdp_miwae_cv: call miwae_cv function on IHDP data


./experiments
|  exp_template.py: template for calling exp_miwae function with different sets of parameters 
|                   (for data generation and for miwae)
|  exp_mi.py: calling exp_mi function with different sets of data generation parameters
|  exp_cevae.py: calling exp_cevae function with different sets of data generation parameters
|  exp_imke_*: additional calls of exp_miwae with more parameter configurations
|  ihdp_miwae_*.py: calling ihdp_miwae function with different sets of parameters
|                   (for missing data generation and for miwae)
|  ihdp_mi.py
|  ihdp_cevae.py

./plots
|
|

./R
|
|

./sandbox
|
|
```


### Running exp:  
$ screen -S exp_7

reserve cpu nb 0 to 23 and set the niceness (for server):
$ taskset -c 0-23 nice -5 python exp_expname.py


### get back the results by scp:

```bash
$ scp tschmitt@drago:/home/tao/tschmitt/miss-vae/results/expname.csv /home/thomas/Documents/miss-vae/results
```


### Running cevae model :  

```bash
conda config --append channels anaconda 
conda config --append channels conda-forge
conda create -n cevae_env python numpy pandas joblib scikit-learn==0.18.1  tensorflow==1.1.0 progressbar==2.3 pip scipy 
conda activate cevae_env
pip install edward==1.3.1 --user

taskset -c 0-23 nice -5 python2 exp_cevae_name.py
```

#### Other issues:
pip3 issues, had to re-install : 
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python3 get-pip.py --force-reinstall

