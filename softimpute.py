# from https://github.com/iskandr/fancyimpute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array

import warnings

F32PREC = np.finfo(np.float32).eps


def soft_impute_rank(X_obs, n_folds = 5, max_rank = 10):

    l_mae = []
    for ii in range(n_folds):
        obs_mask = ~np.isnan(X_obs)
        # randomly sample some test mask
        test_mask = np.array(np.random.binomial(np.ones_like(obs_mask), obs_mask * .2), dtype=bool)

        X_obs_train = X_obs.copy()
        X_obs_train[test_mask] = np.nan

        si = SoftImpute(max_rank=max_rank, verbose=False)
        X_obs_imp = si.fit_transform(X_obs_train)
        si.U
        si.shrinkage_value
        mae_obs = si.mae_obs
        mae_test = np.mean(np.abs(X_obs[test_mask] - X_obs_imp[test_mask]))

        l_mae.append(mae_test)
     
    return si.U, l_mae

#@memory.cache
def get_U_softimpute(X_obs, list_rank=None, boxplot=False, n_folds=3):
    """ return U_hat with SoftImput strategy
    
    Rank are cross selected (wrt MSE) in list_rank"""
    
    #assert np.sum(np.isnan(X_obs)) > 0, 'X_obs do not contains any nan in "get_U_softimpute"'
    
    best_mae = float('inf')
    best_U = None
    best_rank = None
    
    if list_rank is None:
        list_rank = [1,2,3,4,5,6,7,8,9,10,20,30,100, X_obs.shape[1]]
    ll_mae = []
    for max_rank in list_rank:
        if max_rank <= X_obs.shape[1]:
            U, l_mae = soft_impute_rank(X_obs, n_folds = n_folds, max_rank = max_rank)
            ll_mae.append(l_mae)
            
            if np.mean(l_mae) < best_mae:
                best_mae = np.mean(l_mae)
                best_U = U
                best_rank = max_rank
    
    if boxplot:
        sns.swarmplot(data=np.array(ll_mae).T)
        plt.xticks(ticks = np.arange(len(list_rank)), labels=[str(x) for x in list_rank])
        plt.xlabel('SVD rank')
        plt.ylabel('MAE on test fold')
        plt.show()
        
        print('-get_U_softimpute, best_rank=',best_rank)
        print('-get_U_softimpute, best_mae=',best_mae)

    return best_U, best_rank



def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


def generate_random_column_samples(column):
    col_mask = np.isnan(column)
    n_missing = np.sum(col_mask)
    if n_missing == len(column):
        # logging.warn("No observed values in column")
        return np.zeros_like(column)

    mean = np.nanmean(column)
    std = np.nanstd(column)

    if np.isclose(std, 0):
        return np.array([mean] * n_missing)
    else:
        return np.random.randn(n_missing) * std + mean




class Solver(object):
    def __init__(
            self,
            fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None):
        self.fill_method = fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer


    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            warnings.simplefilter("always")
            warnings.warn("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def fill(
            self,
            X,
            missing_mask,
            fill_method=None,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries
        missing_mask : np.array
            Boolean array indicating where NaN entries are
        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column
        inplace : bool
            Modify matrix or fill a copy
        """
        X = check_array(X, force_all_finite=False)

        if not inplace:
            X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = check_array(X, force_all_finite=False)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def project_result(self, X):
        """
        First undo normalization and then clip to the user-specified min/max
        range.
        """
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)

    def solve(self, X, missing_mask):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        X_filled = self.fill(X, missing_mask, inplace=True)
        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                "Expected %s.fill() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_filled)))

        X_result = self.solve(X_filled, missing_mask)
        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                "Expected %s.solve() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_result)))

        X_result = self.project_result(X=X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def fit(self, X, y=None):
        """
        Fit the imputer on input `X`.
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.fit not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only fit_transform is "
            "supported at this time." % (
                self.__class__.__name__,))

    def transform(self, X, y=None):
        """
        Transform input `X`.
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.transform not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only %s.fit_transform is "
            "supported at this time." % (
                self.__class__.__name__, self.__class__.__name__))


class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_value=None,
            convergence_threshold=0.001,
            max_iters=100,
            max_rank=None,
            n_power_iterations=1,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 100.

        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.

        max_iters : int
            Maximum number of SVD iterations

        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.

        n_power_iterations : int
            Number of power iterations to perform with randomized SVD

        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.

        min_value : float
            Smallest allowable value in the solution

        max_value : float
            Largest allowable value in the solution

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
            Print debugging info
        """
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose

        self.U = None
        self.V = None
        self.S = None
        self.mae_obs = None


    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
        # edge cases
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value, max_rank=None):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))

        self.U, self.S, self.V = U_thresh, S_thresh, V_thresh 
        return X_reconstruction, rank

    def _max_singular_value(self, X_filled):
        # quick decomposition of X_filled into rank-1 SVD
        _, s, _ = randomized_svd(
            X_filled,
            1,
            n_iter=5)
        return s[0]

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        X_init = X.copy()

        X_filled = X
        observed_mask = ~missing_mask
        max_singular_value = self._max_singular_value(X_filled)
        if self.verbose:
            print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if self.shrinkage_value:
            shrinkage_value = self.shrinkage_value
        else:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            shrinkage_value = max_singular_value / 50.0

        for i in range(self.max_iters):
            X_reconstruction, rank = self._svd_step(
                X_filled,
                shrinkage_value,
                max_rank=self.max_rank)
            X_reconstruction = self.clip(X_reconstruction)

            # print error on observed data
            if self.verbose:
                mae = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f rank=%d" % (
                        i + 1,
                        mae,
                        rank))

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break
        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))

            
        self.mae_obs = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask)

        return X_filled


if __name__=='__main__':

    from generate_data import gen_lrmf, gen_dlvm
    from generate_data import ampute
    import matplotlib.pyplot as plt
    import seaborn as sns

    Z, X, w, y, ps = gen_lrmf(d = 3)
    X_obs = ampute(X)

    print('boxplot of get_U_softimpute with gen_lrmf(d=3)')
    U = get_U_softimpute(X_obs)