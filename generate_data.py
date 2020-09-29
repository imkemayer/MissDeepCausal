import numpy as np

# X are nonlinear copies of Z obtained via exp and trigonometric functions
def gen_nonlin_copies(n=1000, d=3, p=100, tau=1, link="linear",
                      citcio=False, prop_miss=0,
                      seed=0,
                      x_snr=2., y_snr=2.,
                      mu_z=0, sig_z=1):
    x_sd = 1./(x_snr * np.sqrt(n*p))
    np.random.seed(seed)
    Z = mu_z + sig_z*np.random.randn(n, d)
    X = np.empty([n, p])

    p_over_2 = int(np.ceil(p/2))
    for j in range(0, p_over_2, d):
        X[:,j:min(j+d, p_over_2)] = np.log(np.abs(Z[:,:min(d, p_over_2-j)])) + x_sd*np.random.randn(n, min(d, p_over_2-j))

    for j in range(p_over_2, p, d):
        X[:,j:min(j+d, p)] = 0.5*np.sin(Z[:,:min(d, p-j)]) + 0.5*np.tanh(Z[:,:min(d, p-j)]) + x_sd*np.random.randn(n, min(d, p-j))


    if not(citcio):
        # generate treatment assignment W
        ps, w = gen_treat(Z, link)
        # generate outcome
        y, mu0, mu1 = gen_outcome(Z, w, tau, link, y_snr)
    else:
        ps, w, y, mu0, mu1 = citcio_treat_out(X, prop_miss, seed, link, tau, y_snr)
    assert y.shape == (n, )
    assert w.shape == (n, )

    return Z, X, w, y, ps, mu0, mu1


# Low rank matrix factorization
def gen_lrmf(n = 1000, d = 3, p = 100, tau = 1, link = "linear",
             citcio = False, prop_miss = 0,
             seed = 0, x_snr = 2., y_snr = 2.):
    x_sd = 1./(x_snr * np.sqrt(n*p))
    # V is fixed throughout replications for given fixed n, p, d
    np.random.seed(0)
    V = np.random.randn(p, d)

    np.random.seed(seed)
    Z = np.random.randn(n, d)
    X = Z.dot(V.transpose())
    assert X.shape == (n, p)
    X = X + x_sd*np.random.randn(n, p) # add perturbation to observation matrix

    if not(citcio):
        # generate treatment assignment W
        ps, w = gen_treat(Z, link)
        # generate outcome
        y, mu0, mu1 = gen_outcome(Z, w, tau, link, y_snr)
    else:
        ps, w, y, mu0, mu1 = citcio_treat_out(X, prop_miss, seed, link, tau, y_snr)
    assert y.shape == (n, )
    assert w.shape == (n, )

    return Z, X, w, y, ps, mu0, mu1

# Deep Latent Variable Model (here, we use an MLP)
def gen_dlvm(n = 1000, d = 3, p = 100, tau = 1, link = "linear",
             citcio = False, prop_miss = 0,
             seed = 0,
             h = 5, y_snr = 2.,
             mu_z=0, sig_z=1):

    # V, W, a, b, alpha, beta are fixed throughout replications for fixed n, p, d, h
    np.random.seed(0)
    V = np.random.randn(p, h)
    W = np.random.uniform(0, 1, int(h*d)).reshape((h, d))
    a = np.random.uniform(0, 1, h).reshape((h, 1))
    b = np.random.randn(p, 1)
    alpha = np.random.randn(h, 1)
    beta = np.random.uniform(0, 1, 1)

    np.random.seed(seed)
    Z = mu_z + sig_z*np.random.randn(n, d)
    X = np.empty([n, p])
    for i in range(n):
        mu, Sigma = get_dlvm_params(Z[i,:].reshape(d, 1), V, W, a, b, alpha, beta)
        X[i,:] = np.random.multivariate_normal(mu, Sigma, 1)
    assert X.shape == (n, p)

    if not(citcio):
        # generate treatment assignment W
        ps, w = gen_treat(Z, link)
        # generate outcome
        y, mu0, mu1 = gen_outcome(Z, w, tau, link, y_snr)
    else:
        ps, w, y, mu0, mu1 = citcio_treat_out(X, prop_miss, seed, link, tau, y_snr)
    assert y.shape == (n, )
    assert w.shape == (n, )

    return Z, X, w, y, ps, mu0, mu1

# Compute expectation and covariance of conditional distribution X given Z
def get_dlvm_params(z, V, W, a, b, alpha, beta):
    hu = (W.dot(z) + a).reshape(-1, 1) # same shape as a (not h)
    mu = (V.dot(np.tanh(hu)) + b).reshape(-1, )
    sig = np.exp(alpha.transpose().dot(np.tanh(hu)) + beta)
    Sigma = sig*np.identity(mu.shape[0])

    return mu, Sigma

def citcio_treat_out(X, prop_miss, seed, link, tau, snr):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)
    imp = IterativeImputer()
    X_imp = imp.fit_transform(X_miss)
    ps, w = gen_treat(X_imp, link = link)
    y, mu0, mu1 = gen_outcome(X_imp, w, tau, link, snr)

    return ps, w, y, mu0, mu1

# Generate treatment assignment using confounders Z
def gen_treat(Z, link = "linear"):
    if link == "linear":
        ncolZ = Z.shape[1]
        beta = np.tile([0.6, -0.6], int(np.ceil(ncolZ/2.))) * 2
        beta = beta[:ncolZ]
        f_Z = Z.dot(beta)
        ps = 1/(1+np.exp(-f_Z))
        w = np.random.binomial(1, ps)
        balanced = np.mean(w) > 0.4 and np.mean(w) < 0.6

        # adjust the intercept term if necessary to ensure balanced treatment groups
        offsets = np.linspace(-5, 5, num = 50)
        i, best_idx, min_diff = 0, 0, Z.shape[0]
        while i < len(offsets) and not balanced:
            ps = 1/(1+np.exp(-offsets[i] - f_Z))
            w = np.random.binomial(1, ps)
            balanced = np.mean(w) > 0.4 and np.mean(w) < 0.6
            diff = abs(np.mean(w) - np.mean(1-w))
            if diff < min_diff:
                best_idx, min_diff = i, diff
            i += 1
        if (i == len(offsets)):
            ps = 1/(1+np.exp(-offsets[best_idx]-f_Z))
            w = np.random.binomial(1, ps)
    elif link == "nonlinear":
        raise NotImplementedError("Nonlinear w~Z not defined yet.")
    else:
        raise ValueError("'link' should be choosed between linear and nonlinear model for w. got %s", link)
    return ps, w

# Generate outcomes using confounders Z, treatment assignment w and ATE tau
def gen_outcome(Z, w, tau, link = "linear", snr = 2.):
    if link == "linear":
        n = Z.shape[0]
        ncolZ = Z.shape[1]
        beta = np.tile([-0.2, 0.155, 0.5, -1, 0.2], int(np.ceil(ncolZ/5.)))
        beta = beta[:ncolZ]
        Zbeta = Z.dot(beta).reshape((-1))
        sd = np.sqrt(np.sum(Zbeta**2))/(1.*snr)
        epsilon = sd*np.random.randn(n)
        y = 0.5 + Zbeta + tau*w + epsilon
        mu0 = 0.5 + Zbeta
        mu1 = 0.5 + Zbeta + tau
    elif link == "nonlinear":
        raise NotImplementedError("Nonlinear w~Z not defined yet.")
    else:
        raise ValueError("'link' should be choosed between linear and nonlinear model for y. got %s", link)
    return y, mu0, mu1

# Generate missing values in X such that, on average, X contains n*p*prop_miss missing values
def ampute(X, prop_miss = 0.1, seed = 0):
    np.random.seed(seed)
    n, p = X.shape
    # initialize mask with 1 (assume that X is fully observed)
    save_mask = ~np.isnan(X)
    # ensure that X contains at least 1 entry per row (assume also that X.shape[0] > X.shape[1])
    for i in range(X.shape[0]):
      j = np.random.choice(X.shape[1], 1)
      save_mask[i, j] = 0
    mask = np.array(np.random.binomial(np.ones_like(save_mask), save_mask * np.minimum(1., prop_miss*(p+1.)/p)), dtype=bool)
    X_miss = X.copy()
    X_miss[mask] = np.nan
    return X_miss
