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
|    - miwae: MIWAE adapted from P.-A. Mattei with additional sampling from Z|X
|             using self-normalized importance sampling weights
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
|  mdc.py
|    - exp_complete: ATE estimation on complete data (X or Z) using estimators from estimators.py
|    - exp_mean: ATE estimation on mean-imputed data using estimators from estimators.py
|    - exp_mf: ATE estimation on approximate latent factors obtained with SoftImpute using estimators from estimators.py
|    - exp_mi: ATE estimation via multiple imputation using estimators from estimators.py
|    - exp_mdc: ATE estimation via MDC (process and mi) using estimators from estimators.py
|
|
|  softimpute_cv.py:
|   - softimpute: Softimpute on incomplete matrix with given regularization parameter lamb
|   - cv_softimpute: Returns cross-validation error of softimpute on a grid of values of lamb
|
|
|  exp_template.py: Template for calling all functions of main.py on data
|                   generated with different sets of parameters (for data
|                   generation, for mdc and for mi).
|                   Parameters for the data generation and models are passed
|                   through flags, run `python exp_template.py --help` to have
|                   a full description. By default it executes all possible
|                   parameters.
|
|
|  plot_results.rmd: Once the experiments are run and the results stored in a csv
|                    file in the results folder, this R Markdown allows to re-produce
|                    all Figures from the paper and the supplementary material.

./results
|  *.csv: Experiment results in form of csv files (containing output of exp_template.py) are saved here
|
```

Note that treatment assignment vector is assumed to have values in {0, 1} (not {-1, 1}).
