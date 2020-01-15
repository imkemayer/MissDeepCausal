import numpy as np

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate() # This needs to be called
base = importr('base')
utils = importr('utils')
softImpute = importr('softImpute')

def cross_valid(x, r, warm, nfolds = 3):
	from sklearn.model_selection import KFold

	kfn = KFold(n_splits = nfolds) 
	kfp = KFold(n_splits = nfolds) 

	kfn_fold_train = []
	kfn_fold_test = []
	for fold_train, fold_test in kfn.split(np.arange(x.shape[0])):
		kfn_fold_train.append(fold_train)
		kfn_fold_test.append(fold_test)
	kfp_fold_train = []
	kfp_fold_test = []
	for fold_train, fold_test in kfp.split(np.arange(x.shape[1])):
		kfp_fold_train.append(fold_train)
		kfp_fold_test.append(fold_test)

	error_folds = []
	fit_folds = []
	for k in range(nfolds):
		temp_data = x.copy()
		temp_data[kfn_fold_test[k]][:, kfp_fold_test[k]] = np.nan 
		if warm is None:
			fit = softImpute.softImpute(temp_data, rank_max = r, type = "als", 
										maxit = 1000)
		else:
			fit = softImpute.softImpute(temp_data, rank_max = r, type = "als", 
										maxit = 1000, warm_start = warm)
		temp_data_test = x[kfn_fold_test[k]][:, kfp_fold_test[k]] 

		pred = softImpute.impute(fit, i = base.rep(kfn_fold_test[k], len(kfp_fold_test[k])), j = base.rep(kfp_fold_test[k], each=len(kfn_fold_test[k])))

		error = np.nanmean((x[kfn_fold_test[k]][:, kfp_fold_test[k]].flatten('F') - pred)**2)
		error_folds.append(error)
		fit_folds.append(fit)

	result = dict()
	result['error'] = np.mean(error_folds)
	result['fit'] = fit_folds[np.argmin(error_folds)]
	return result


def recover_pca_gaussian_cv(x, r_seq = None, nfolds = 3):
	"""
	Gaussian matrix factorization on the noisy proxy matrix
	with rank chosen by cross validation from r_seq 
	the matrix factorization is carried out by the softImpute R package  
	"""
	if r_seq is None:
		r_seq = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]

	cv_error = []
	warm_list = []
	for r in range(len(r_seq)):
		if r == 0:
			if r_seq[r] < x.shape[1]:
				temp = cross_valid(x, r_seq[r], warm = None, nfolds = nfolds)
				cv_error.append(temp['error'])
				warm_list.append(temp['fit'])
		else:
			if r_seq[r] < x.shape[1]:
				temp = cross_valid(x, r_seq[r], warm = warm_list[r-1], nfolds = nfolds)
				cv_error.append(temp['error'])
				warm_list.append(temp['fit'])
	best_r = r_seq[np.argmin(cv_error)]
	warm = warm_list[np.argmin(cv_error)]
	
	result = softImpute.softImpute(x, rank_max = best_r, type = "als", maxit = 1000, warm_start = warm)
	UhatD = np.dot(np.array(result[0]).reshape((x.shape[0], best_r)), np.diag(np.array(result[1])))
	return UhatD, best_r


