import math, re
import numpy as np
import pandas as pd

from scipy.stats import genextreme, pearson3, gumbel_r, lognorm, norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import percentileofscore, skew
from scipy.optimize import minimize
from scipy.special import gamma, gammaln

from tqdm import tqdm

from hydrodiy.data.containers import Vector

DISTRIBUTION_NAMES = ["Normal", "Empirical", "GEV", "LogPearson3", \
                        "Gumbel", "LogNormal"]

EULER_CONSTANT = 0.577215664901532

MAXLIKE_MINIMIZE_KWARGS = {
    "Nelder-Mead": {
        "disp": False, \
        "xatol": 1e-7, \
        "fatol": 1e-7, \
        "maxfev": 10000, \
        "maxiter": 10000
    }, \
    "Powell": {
        "disp": False, \
        "xtol": 1e-7, \
        "ftol": 1e-7, \
        "maxfev": 10000, \
        "maxiter": 10000
    },
    "BFGS": {
        "disp": False, \
        "gtol": 1e-7, \
        "eps": 1e-4, \
        "maxiter": 10000
    }
}

def _prepare(data):
    if data.ndim == 1:
        data = np.array(data)[:, None]
    _, nvars = data.shape

    data[np.isnan(data)] = -2e100
    data_sorted = np.sort(data, axis=0)
    data_sorted[data_sorted<-1e100] = np.nan
    nval = (~np.isnan(data_sorted)).sum(axis=0)[None, :]
    nval = nval.astype(np.int64)

    errmsg = f"Expected length of valid data>=4, got {nval}."
    assert np.all(nval>=4), errmsg
    return data_sorted, nval, nvars


def _check_1d_data(data):
    errmsg = f"Expected data to be 1 dimensional, got data.ndim={data.ndim}."
    assert data.ndim == 1, errmsg
    errmsg = "Expected no nan in data."
    assert np.all(~np.isnan(data)), errmsg


def _check_censoring_data(data, censoring_status, censoring_threshold):
    """ Censoring status:
            1: Observed
            2: Not observed, below censor
            3: Not obserbed, above censor
    """
    if censoring_status is None:
        censoring_status = np.ones(len(data), dtype=int)
    if censoring_threshold is None:
        censoring_threshold = -np.inf

    censoring_status = censoring_status.astype(int)
    errmsg = "Expected len(data)==len(censoring_status)."
    assert len(data) == len(censoring_status), errmsg

    # Check no censored data above threshold
    idx = data>censoring_threshold
    errmsg = "Some data are marked as censored, but they are above censoring threshold."
    assert np.sum(idx & (censoring_status!=1)) == 0, errmsg

    # Check no valid data below threshold
    idx = data<censoring_threshold
    errmsg = "Some data are not marked as censored, but they are below censoring threshold."
    assert np.sum(idx & (censoring_status==1)) == 0, errmsg

    iobs = censoring_status==1
    ncens = [np.sum(censoring_status==i) for i in range(1, 4)]

    return iobs, ncens


def _minimize_fun(objfun, theta0, methods):
    opt, fmin = None, np.inf
    for method in methods:
        assert method in MAXLIKE_MINIMIZE_KWARGS
        options = MAXLIKE_MINIMIZE_KWARGS[method]
        o = minimize(objfun, theta0, method=method, options=options)
        if o.fun < fmin:
            opt = o
            opt.method = method
            fmin = o.fun

    if opt is None:
        errmsg = "Could not minimize function."
        raise ValueError(errmsg)

    return opt


def _comb(n, i):
    """ Function much faster than scipy.comb """
    if i>n:
        return 0
    elif i==0:
        return 1
    elif i==1:
        return n
    elif i==2:
        return n*(n-1)/2
    elif i==3:
        return n*(n-1)*(n-2)/6
    elif i==4:
        return n*(n-1)*(n-2)*(n-3)/24
    elif i==5:
        return n*(n-1)*(n-2)*(n-3)*(n-4)/120
    elif i==6:
        return n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)/720
    elif i==7:
        return n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)*(n-6)/5040
    elif i==8:
        return n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)*(n-6)*(n-7)/40320


def lh_moments(data, eta=0, compute_lam4=True):
    """Compute LH moments as per Wang et al. (1997).

    Parameters
    ----------
    data : numpy.ndarray
        1D vector containing data.
    eta : int
        Shift parameter. eta=0 lead to no eta,
        i.e. computing L moments.
    compute_lam4 : bool
        Compute lam4 or not.

    Returns
    lam1 : float
        First LH moment.
    lam2 : float
        Second LH moment.
    lam3 : float
        Third LH moment.
    lam4 : float
        Fourth LH moment.
    """
    # Prepare data
    data_sorted, nval, nvars = _prepare(data)
    eta = int(eta)

    errmsg = "Expected same number of valid data for all variables."
    assert np.max(np.abs(nval-nval[0, 0]))<1e-10, errmsg
    nval = nval[0, 0]

    # Compute L moments, see ARR, book 3 page 47
    z = np.zeros(nvars)
    lam1, lam2, lam3, lam4 = z.copy(), z.copy(), z.copy(), z.copy()
    v1, v2, v3, v4 = z.copy(), z.copy(), z.copy(), z.copy()

    for i in range(1, nval+1):
        # Compute C factors (save time by reusing data)
        Cim1e0 = _comb(i-1, eta)
        Cim1e1 = _comb(i-1, eta+1)
        Cim1e2 = _comb(i-1, eta+2)

        Cnmi1 = nval-i
        Cnmi2 = _comb(nval-i, 2)
        Cnmi3 = _comb(nval-i, 3)

        # Compute components of moments
        d = data_sorted[i-1, :]
        v1 += Cim1e0*d
        v2 += (Cim1e1-Cim1e0*Cnmi1)*d
        v3 += (Cim1e2-2*Cim1e1*Cnmi1+Cim1e0*Cnmi2)*d

        if compute_lam4:
            v4 += (_comb(i-1, eta+3)-3*Cim1e2*Cnmi1\
                            +3*Cim1e1*Cnmi2-Cim1e0*Cnmi3)*d

    lam1 = v1/_comb(nval, eta+1)
    lam2 = v2/_comb(nval, eta+2)/2
    lam3 = v3/_comb(nval, eta+3)/3
    if compute_lam4:
        lam4 = v4/_comb(nval, eta+4)/4
    else:
        lam4 = np.nan*np.zeros(nvars)

    return lam1, lam2, lam3, lam4


def factory(distname):
    txt = "/".join(DISTRIBUTION_NAMES)
    errmsg = f"Expected distnames in {txt}, got {distname}."
    assert distname in DISTRIBUTION_NAMES, errmsg

    if distname == "GEV":
        return GEV()
    elif distname == "Empirical":
        return Empirical()
    elif distname == "LogPearson3":
        return LogPearson3()
    elif distname == "Gumbel":
        return Gumbel()
    elif distname == "LogNormal":
        return LogNormal()
    elif distname == "Normal":
        return Normal()
    else:
        raise ValueError(errmsg)


class FloodFreqDistribution():
    """ Base class for flood frequency distribution """

    def __init__(self, name, params):
        self.name = name
        self._params = params

        # Uninformative prior
        nparams = params.nval
        self._neglogprior_mvn_mu = params.defaults.copy()
        self._neglogprior_mvn_Sigma = 1e100*np.eye(nparams)

    def __getattribute__(self, name):
        # Except certain names to avoid infinite recursion
        if name in ["name", "_params"]:
            return super(FloodFreqDistribution, self).__getattribute__(name)

        if name in self._params.names:
            return getattr(self._params, name)

        if name.startswith("neglogprior_mvn"):
            return getattr(self, f"_{name}")

        return super(FloodFreqDistribution, self).__getattribute__(name)

    def __setattr__(self, name, value):
        # Except certain names to avoid infinite recursion
        if name in ["name", "_params"]:
            super(FloodFreqDistribution, self).__setattr__(name, value)
            return

        if name in self._params.names:
            return setattr(self._params, name, value)

        super(FloodFreqDistribution, self).__setattr__(name, value)


    def __setitem__(self, key, value):
        if not key in self.params.names:
            txt = "/".join(list(self.params.names))
            raise ValueError(f"Expected {key} in {txt}.")
        setattr(self, key, value)


    def __getitem__(self, key):
        if not key in self.params.names:
            txt = "/".join(list(self.params.names))
            raise ValueError(f"Expected {key} in {txt}.")
        return getattr(self, key)


    def __str__(self):
        txt = f"{self.name} flood frequency distribution:\n"
        txt += f"{' '*2}Params\n"

        pmu = self.neglogprior_mvn_mu
        psig = np.sqrt(np.diag(self.neglogprior_mvn_Sigma))

        for iparam, (pn, v) in enumerate(self.params.to_series().items()):
            txt += f"{' '*4}{pn:9s} = {v:8.3f} "
            txt += f"(prior {pmu[iparam]:0.1f} +/- {psig[iparam]:2.1e})"
            txt += "\n"
        try:
            x0, x1 = self.support
            txt += f"\n{' '*2}Support   = [{x0:0.3f}, {x1:0.3f}]\n"
        except NotImplementedError:
            txt += "\n"
        return txt

    @classmethod
    def from_dict(cls, dd):
        ffd = factory(dd["name"])
        ffd._neglogprior_mvn_mu = np.array(dd["neglogprior_mvn_mu"])
        ffd._neglogprior_mvn_Sigma = np.array(dd["neglogprior_mvn_Sigma"])

        for pn, v in dd["params"].items():
            ffd.params[pn] = v

        return ffd

    def to_dict(self):
        dd = {"name": self.name, "params":{}, \
                "config":{}, "params_archive": {}, \
                "neglogprior_mvn_mu": self.neglogprior_mvn_mu.tolist(),\
                "neglogprior_mvn_Sigma": self.neglogprior_mvn_Sigma.tolist()
        }
        for pn, v in self.params.to_series().items():
            dd["params"][pn] = v

        return dd

    @property
    def params(self):
        return self._params

    def clone(self):
        return FloodFreqDistribution.from_dict(self.to_dict())

    def set_dict_params(self, params):
        for pn in self.params.names:
            self[pn] = params[pn]

    def params2txt(self):
        txt = ""
        for pn, v in self.params.to_series().items():
            txt += f" {pn}={v:0.3f}"
        return txt

    def get_scipy_params(self, params):
        errmsg = f"Method get_scipy_params not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def fit_lh_moments(self, data, eta=0):
        errmsg = f"Method fit_lh_moment not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def logpdf(self, x):
        errmsg = f"Method logpdf not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def pdf(self, x):
        errmsg = f"Method pdf not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def cdf(self, x):
        errmsg = f"Method cdf not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def logcdf(self, x):
        errmsg = f"Method logcdf not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def ppf(self, q):
        errmsg = f"Method ppf not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def ppfvect(self, q, samples):
        """ Vectorized ppf function. """
        errmsg = f"Method ppfvect not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def rvs(self, size):
        errmsg = f"Method rvs not implemented for class {self.name}."
        raise NotImplementedError(errmsg)


    def rvs_truncated(self, size, trunc_l=-np.inf, trunc_r=np.inf):
        """ Truncated random sampling in [trunc_l, trunc_r]. """
        x0, x1 = self.support
        trunc_l = max(x0, trunc_l)
        q_l = self.cdf(trunc_l)
        trunc_r = min(x1, trunc_r)
        q_r = self.cdf(trunc_r)

        u = np.random.uniform(q_l, q_r, size=size)
        return self.ppf(u)

    @property
    def support(self):
        errmsg = f"Property support not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def return_period(self, T):
        return self.ppf(1-1/np.array(T))


    def params_guess(self, data):
        """ Provide a simple estimate of model params to fit data."""
        errmsg = f"Method params_guess not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def get_negloglike(self, data, censoring_status=None, censoring_threshold=None):
        """ negative log-likelihood taking into account censored information.
            censoring status =
            1: Observed
            2: Not observed, below censor
            3: Not observed, above censor
        """
        # Check data
        _check_1d_data(data)
        iobs, ncens = _check_censoring_data(data, censoring_status, censoring_threshold)

        # Build neg log like
        maxs = self.params.maxs
        mins = self.params.mins
        data_obs = np.array(data[iobs])

        def fun(theta):
            if not isinstance(theta, np.ndarray):
                theta = np.array(theta)

            # Set params
            if np.any((maxs-theta<0)|(mins-theta>0)|np.isnan(theta)):
                return np.inf

            # use the underscore to bypass checks
            # in params object
            self.params._values = theta

            # Observed data
            nll = 0.
            n1, n2, n3 = ncens
            if n1>0:
                nll -= self.logpdf(data_obs).sum()

            # Not observed - below censoring threshold
            if n2>0:
                nll -= self.logcdf(censoring_threshold)*n2

            # Not observed - above censoring threshold
            if n3>0:
                nll -= np.log(1-self.cdf(censoring_threshold))*n3

            if not np.isfinite(nll):
                return np.inf

            return nll

        # Test run
        self.params_guess(data_obs)
        theta = self.params.values
        f1 = fun(theta)
        f2 = fun(theta+1e-3)
        if np.isfinite(f1) and np.isfinite(f2):
            errmsg = "Log likelihood function seems to be insensitive to parameter values."
            assert abs(f1-f2)>0, errmsg

        return fun


    def uninformative_neglogprior(self, theta):
        errmsg = f"Method uninformative_neglogprior not implemented for class {self.name}."
        raise NotImplementedError(errmsg)


    def get_neglogprior(self, uninformative=False):
        if uninformative:
            def fun(theta):
                return self.uninformative_neglogprior(theta)
        else:
            def fun(theta):
                return -mvn.logpdf(theta, mean=self.neglogprior_mvn_mu, \
                                    cov=self.neglogprior_mvn_Sigma)
        # Test run
        fun(self.params.defaults)

        # Return function
        return fun


    def get_neglogposterior(self, data, uninformative_prior=False, \
                                            censoring_status=None, \
                                            censoring_threshold=None):
        neglogp = self.get_neglogprior(uninformative_prior)
        negloglike = self.get_negloglike(data, censoring_status, \
                                        censoring_threshold)
        def fun(theta):
            nllike = negloglike(theta)
            nlprior = neglogp(theta)
            return nllike+nlprior

        # Return function
        return fun


    def maximise(self, data, maxtype="likelihood", \
                        censoring_status=None, censoring_threshold=None, \
                        methods=["Nelder-Mead", "Powell"], theta0=None, \
                        uninformative_prior=False):
        """ Estimate maximum likelihood parameters. Censored data can be
        defined via the censoring variables.

        Parameters
        ----------
        data : numpy.ndarray
            1D array containing data.
        maxtype : str
            Indicate if we search for maximum likelihood ('likelihood')
            or posterior ('posterior').
        censoring_status : numpy.ndarray
            1D array of same length than data containing the censoring
            status of each data point:
            1 = observed.
            2 = not observed, below censoring threshold.
            3 = not observed, above censoring threshold.
        censoring_threshold : float
            Threshold used when censoring_status is either 2 or 3.
         method : str
            Optimisation method. Set to [Nelder-Mead, Powell]
            by default, i.e. use all methods and retain the best
            result.
        theta0 : numpy.ndarray
            Starting point.
        uninformative_prior : bool
            Use uninformative prior if required.
            Only used if maxtype='posterior'.

        Returns
        -------
        final_simplex : tuple
            Final simplex returned by Nelder-Mead. Can be used to
            estimate Hessian. The first component of the tuple are
            point coordinates and the second are negative likelihood
            values.
        """
        if maxtype == "likelihood":
            neglogfun = self.get_negloglike(data, censoring_status, \
                                            censoring_threshold)
        elif maxtype == "posterior":
            neglogfun = self.get_neglogposterior(data, \
                                    censoring_status=censoring_status, \
                                    censoring_threshold=censoring_threshold, \
                                    uninformative_prior=uninformative_prior)
        # Define starting point
        self.params_guess(data)
        if theta0 is None:
            theta0 = self.params.values

        # Run minimizer
        opt = _minimize_fun(neglogfun, theta0, methods)
        self.params.values = opt.x
        return opt


    def samples2expon(self, samples):
        """ Transform log transformed parameters to original space."""
        assert isinstance(samples, pd.DataFrame)
        expon = samples.apply(lambda x: np.exp(x) if x.name.startswith("log") else x)
        expon.columns = [re.sub("^log", "", cn) for cn in expon.columns]
        return expon


    def expon2samples(self, expon):
        """ Transform parameters from original space to log transform."""
        assert isinstance(expon, pd.DataFrame)
        trans = {pn: pn.startswith("log")  for pn in self.params.names}
        samples = expon.apply(lambda x: np.log(x) \
                                if trans.get(f"log{x.name}", False) else x)
        samples.columns = [f"log{cn}" if trans.get(f"log{cn}", False) else cn \
                                for cn in expon.columns]
        return samples

    def samples2scipy(self, samples):
        """ Transform parameters to scipy rvs function parameters. """
        sparams = []
        for i, sample in samples.iterrows():
            sparams.append(self.get_scipy_params(sample))

        return pd.DataFrame(sparams)




class Normal(FloodFreqDistribution):
    def __init__(self):
        name = "Normal"
        params = Vector(["logsig", "mu"], [0, 0], [-np.inf]*2, [np.inf]*2)
        super(Normal, self).__init__(name, params)

    def get_scipy_params(self, params):
        return {"loc": params.mu, "scale": math.exp(params.logsig)}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params(self.params)
                f = getattr(norm, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params(self.params)
                return norm.rvs(size=size, **kw)
            return fun

        return super(Normal, self).__getattribute__(name)

    @property
    def sig(self):
        return math.exp(self.logsig)

    @property
    def support(self):
        return -np.inf, np.inf

    def ppfvect(self, q, samples):
        """ Vectorized ppf function. """
        assert q.ndim ==1
        return norm.ppf(q[None, :], loc=samples.mu.values[:, None], \
                                    scale=np.exp(samples.logsig).values[:, None])


    def params_guess(self, data):
        self.mu = np.mean(data)
        self.logsig = math.log(np.std(data, ddof=1))

    def fit_lh_moments(self, data, eta=0):
        errmsg = f"Expected eta=0, got {eta}."
        assert eta==0, errmsg
        lam1, lam2, _, _ = lh_moments(data, eta, compute_lam4=False)
        sqpi = math.sqrt(math.pi)
        return pd.DataFrame({"mu": lam1, "logsig": np.log(lam2*sqpi)})

    def uninformative_neglogprior(self, theta):
        """ p(mu, logsig) ~ 1 for uninformative """
        return 0.


class Empirical(FloodFreqDistribution):
    """ Empirical distribution class"""

    def __init__(self):
        name = "Empirical"
        params = Vector(["noparam"])
        super(Empirical, self).__init__(name, params)
        self._data_sorted = None

    @property
    def has_data(self):
        if self._data_sorted is None:
            raise AttributeError("No data available, please use set_data")
        return True

    @property
    def support(self):
        self.has_data
        return self._data_sorted[0], self._data_sorted[-1]

    def set_data(self, data):
        data_sorted, _, nvars = _prepare(data)
        errmsg = f"Expected one dimensional data, got nvars={nvars}."
        assert nvars == 1, errmsg
        self._data_sorted = data_sorted[:, 0]

    def cdf(self, x):
        self.has_data
        if hasattr(x, "__iter__"):
            return np.array([percentileofscore(self._data_sorted, xx, "weak")/100.\
                        for xx in x])
        else:
            return percentileofscore(self._data_sorted, x, "weak")/100.

    def logcdf(self, x):
        return np.log(self.cdf(x))

    def ppf(self, q):
        self.has_data
        return np.percentile(self._data_sorted, q*100)

    def rvs(self, size):
        self.has_data
        return np.random.choice(self._data_sorted, size)



class GEV(FloodFreqDistribution):
    """ GEV distribution class"""

    def __init__(self):
        name = "GEV"
        params = Vector(["kappa", "logalpha", "tau"], \
                    [0.003, 0., 0.], [-3, -50, -np.inf], \
                    [3, 50, np.inf])
        super(GEV, self).__init__(name, params)

    @property
    def alpha(self):
        return math.exp(self.logalpha)

    @property
    def support(self):
        x0 = self.tau+self.alpha/self.kappa
        if self.kappa<0:
            return x0, np.inf
        else:
            return -np.inf, x0

    def get_scipy_params(self, params):
        return {"c": params.kappa, "loc": params.tau, "scale": math.exp(params.logalpha)}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params(self.params)
                f = getattr(genextreme, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params(self.params)
                return genextreme.rvs(size=size, **kw)
            return fun

        return super(GEV, self).__getattribute__(name)

    def ppfvect(self, q, samples):
        """ Vectorized ppf function. """
        assert q.ndim ==1
        sparams = self.samples2scipy(samples)
        return genextreme.ppf(q[None, :], \
                                c=sparams.loc[:, "c"].values[:, None], \
                                loc=sparams.loc[:, "loc"].values[:, None], \
                                scale=sparams.loc[:, "scale"].values[:, None])

    def params_guess(self, data):
        samples = self.fit_lh_moments(data, 0).iloc[0]
        self.kappa = samples.kappa
        self.logalpha = samples.logalpha
        self.tau = samples.tau

    #def fit_pw_moments(self, data):
    #    beta0, beta1, beta2 = _pw_moments(data)

    #    # GEV fit according to Hoskings et al., 1985
    #    c = (2*beta1-beta0)/(3*beta2-beta0)-math.log(2)/math.log(3)
    #    kappa = 7.8590*c+2.9554*c*c
    #    alpha = (2*beta1-beta0)*kappa/gamma(1+kappa)/(1-2**(-kappa))

    #    self.kappa = kappa
    #    self.logalpha = math.log(alpha)
    #    self.tau = beta0 + alpha*(gamma(1+kappa)-1)/kappa


    def fit_lh_moments(self, data, eta=0):
        """ See Wang et al. (1997). """
        errmsg = f"Expected eta in [0, 4], got {eta}."
        assert eta<=4 and eta>=0, errmsg

        # Get LH moments
        lam1, lam2, lam3, _ = lh_moments(data, eta, compute_lam4=False)
        tau3 = lam3/lam2

        # See Wang et al.(1997), Equation 26
        coefs_all = {
            0: [0.2849, -1.8213, 0.8140, -0.2835],\
            1: [0.4823, -2.1494, 0.7269, -0.2103],\
            2: [0.5914, -2.3351, 0.6442, -0.1616],\
            3: [0.6618, -2.4548, 0.5733, -0.1273],\
            4: [0.7113, -2.5383, 0.5142, -0.1027]
        }
        coefs = coefs_all[eta]
        kappa = coefs[0]+coefs[1]*tau3+coefs[2]*tau3**2+coefs[3]*tau3**3

        # See wang et al. (1997), Equation 19
        g = gamma(1+kappa)
        e = (1+eta)**(-kappa)
        lhs = (eta+2)*g/2/kappa*(-(eta+2)**(-kappa)+e)
        alpha = lam2/lhs

        # See wang et al. (1997), Equation 18
        lhs = alpha/kappa*(1-g*e)
        tau = lam1-lhs

        return pd.DataFrame({"kappa": kappa, "logalpha": np.log(alpha), \
                                "tau": tau})



class LogPearson3(FloodFreqDistribution):
    """ Log Pearson III distribution class"""

    def __init__(self):
        name = "LogPearson3"
        # Bounds for g (Skew) from Griffis and Stedinger (2007)
        params = Vector(["g", "s", "m"], \
                    [0., 1., 0.], \
                    [-5, 1e-5, -np.inf], \
                    [5, np.inf, np.inf])
        super(LogPearson3, self).__init__(name, params)

    def get_scipy_params(self, params):
        return {"skew": params.g, "loc": params.m, "scale": params.s}

    @property
    def alpha(self):
        return 4/self.g**2

    @property
    def beta(self):
        return np.sign(self.g)*math.sqrt(self.alpha)/self.s

    @property
    def tau(self):
        return self.m-self.alpha/self.beta

    @property
    def support(self):
        if self.beta<0:
            return [0, math.exp(max(-100, min(100, self.tau)))]
        else:
            return [math.exp(max(-100, min(100, self.tau))), np.inf]

    def ppfvect(self, q, samples):
        """ Vectorized ppf function. """
        assert q.ndim ==1
        sparams = self.samples2scipy(samples)

        skew, loc, scale = [sparams.loc[:, n].values[:, None] for n in \
                                            ["skew", "loc", "scale"]]
        ipos = skew[:, 0]>0
        ppf = np.zeros((len(samples), len(q)))
        ppf[ipos] = np.exp(pearson3.ppf(q[None, :], \
                            skew=skew[ipos], loc=loc[ipos], \
                            scale=scale[ipos]))
        ppf[~ipos] = np.exp(pearson3.ppf(1-q[None, :], \
                            skew=skew[~ipos], loc=loc[~ipos], \
                            scale=scale[~ipos]))
        return ppf

    def pdf(self, x):
        kw = self.get_scipy_params(self.params)
        return pearson3.pdf(np.log(x), **kw)/x

    def logpdf(self, x):
        kw = self.get_scipy_params(self.params)
        return pearson3.logpdf(np.log(x), **kw)-np.log(x)

    def cdf(self, x):
        kw = self.get_scipy_params(self.params)
        return pearson3.cdf(np.log(x), **kw)

    def logcdf(self, x):
        kw = self.get_scipy_params(self.params)
        return pearson3.logcdf(np.log(x), **kw)

    def ppf(self, q):
        kw = self.get_scipy_params(self.params)
        qq = q if self.g>0 else 1-q
        return np.exp(pearson3.ppf(qq, **kw))

    def params_guess(self, data):
        logx = np.log(data)
        self.g = skew(logx)
        self.m = logx.mean()
        self.s = logx.std(ddof=1)

    def rvs(self, size):
        kw = self.get_scipy_params(self.params)
        return np.exp(pearson3.rvs(size=size, **kw))

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking (1990), http://lib.stat.cmu.edu/general/lmoments """
        errmsg = f"Expected eta=0, got {eta}."
        assert eta==0, errmsg

        # Get LH moments from log transform data
        lx = np.log(data)
        lam1, lam2, lam3, _ = lh_moments(lx, eta, compute_lam4=False)
        tau3 = lam3/lam2

        # initialise
        mu = np.zeros_like(lam1)
        sigma = np.zeros_like(lam1)
        gam = np.zeros_like(lam1)

        # Translation of routine PELPE3
        # See http://lib.stat.cmu.edu/general/lmoments, line 2866
        C1,C2,C3 = 0.2906,  0.1882,  0.0442
        D1,D2,D3 = 0.36067,-0.59567, 0.25361
        D4,D5,D6 =-2.78861, 2.56096,-0.77045
        PI3,ROOTPI = 9.4247780,1.7724539
        SMALL = 1e-15
        # XMOM(1) -> lam1
        # XMOM(2) -> lam2
        # XMOM(3) -> tau3

        T3 = np.abs(tau3)

        idx1 = (lam2<=0) | (T3>=1.)
        mu[idx1] = 0.
        sigma[idx1] = 0.
        gam[idx1] = 0.

        idx2 = (T3<=SMALL) & ~idx1
        mu[idx2] = lam1[idx2]
        sigma[idx2] = lam2[idx2]*ROOTPI
        gam[idx2] = 0.

        idx3 = (T3>=1./3) & ~idx1 & ~idx2
        ALPHA = np.zeros_like(lam1)
        Ti3 = 1.-T3[idx3]
        ALPHA[idx3] = Ti3*(D1+Ti3*(D2+Ti3*D3))/(1.+Ti3*(D4+Ti3*(D5+Ti3*D6)))

        idx4 = ~idx1 & ~idx2 & ~idx3
        Ti4 = PI3*T3[idx4]*T3[idx4]
        ALPHA[idx4] = (1.+C1*Ti4)/(Ti4*(1.+Ti4*(C2+Ti4*C3)))

        RTALPH = np.sqrt(ALPHA)
        BETA = ROOTPI*lam2*np.exp(gammaln(ALPHA)-gammaln(ALPHA+0.5))

        idx5 = idx3 | idx4
        mu[idx5] = lam1[idx5]
        sigma[idx5] = BETA[idx5]*RTALPH[idx5]
        gam[idx5] = 2./RTALPH[idx5]
        gam[idx5 & (tau3<0)] *= -1

        return pd.DataFrame({"g": gam, "s": sigma, "m": mu})



class Gumbel(FloodFreqDistribution):
    """ GEV distribution class"""

    def __init__(self):
        name = "Gumbel"
        params = Vector(["logalpha", "tau"], \
                    [0., 0.], \
                    [-50, -np.inf], \
                    [50, np.inf])
        super(Gumbel, self).__init__(name, params)

    @property
    def alpha(self):
        return math.exp(self.logalpha)

    @property
    def support(self):
        return -np.inf, np.inf

    def get_scipy_params(self, params):
        return {"loc": params.tau, "scale": math.exp(params.logalpha)}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params(self.params)
                f = getattr(gumbel_r, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                return gumbel_r.rvs(size=size, scale=self.alpha, loc=self.tau)
            return fun

        return super(Gumbel, self).__getattribute__(name)

    def ppfvect(self, q, samples):
        """ Vectorized ppf function. """
        assert q.ndim ==1
        sparams = self.samples2scipy(samples)
        return gumbel_r.ppf(q[None, :], \
                            loc=sparams.loc[:, "loc"].values[:, None], \
                            scale=sparams.loc[:, "scale"].values[:, None])

    def fit_lh_moments(self, data, eta=0):
        """ See Wang et al. (1997). """
        errmsg = f"Expected eta in [0, 4], got {eta}."
        assert eta<=4 and eta>=0, errmsg

        # Get LH moments
        lam1, lam2, lam3, _ = lh_moments(data, eta, compute_lam4=False)

        # See wang et al. (1997), Equation 23
        lhs = (eta+2)/2*(math.log(eta+2)-math.log(eta+1))
        alpha = lam2/lhs

        # See wang et al. (1997), Equation 22
        lhs = alpha*(EULER_CONSTANT+math.log(eta+1))
        tau = lam1-lhs

        return pd.DataFrame({"logalpha": np.log(alpha), "tau": tau})

    def params_guess(self, data):
        alpha = data.var()*6/math.pi**2
        self.logalpha = math.log(alpha)
        self.tau = data.mean()-EULER_CONSTANT*alpha



class LogNormal(FloodFreqDistribution):
    """ LogNormal distribution class"""

    def __init__(self):
        name = "LogNormal"
        params = Vector(["s", "m"], \
                    [1., 0.], \
                    [1e-5, -np.inf], \
                    [np.inf, np.inf])
        super(LogNormal, self).__init__(name, params)

    @property
    def support(self):
        return 0, np.inf

    def get_scipy_params(self, params):
        return {"s": params.s, "loc": 0., "scale": math.exp(params.m)}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params(self.params)
                f = getattr(lognorm, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params(self.params)
                return lognorm.rvs(size=size, **kw)
            return fun

        return super(LogNormal, self).__getattribute__(name)

    def ppfvect(self, q, samples):
        assert q.ndim ==1
        sparams = self.samples2scipy(samples)
        return lognorm.ppf(q[None, :], \
                           s=sparams.loc[:, "s"].values[:, None], \
                           loc=sparams.loc[:, "loc"].values[:, None], \
                           scale=sparams.loc[:, "scale"].values[:, None])

    def params_guess(self, data):
        logx = np.log(data)
        self.m = logx.mean()
        self.s = (logx-self.m).std(ddof=1)

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking and Wallis (1997), Appendix, page 198. """
        errmsg = f"Expected eta=0, got {eta}."
        assert eta==0, errmsg

        # Get LH moments
        lx = np.log(data)
        lam1, lam2, _, _ = lh_moments(lx, eta, compute_lam4=False)

        return pd.DataFrame({"s": lam2*math.sqrt(math.pi), \
                            "m": lam1})

    def uninformative_neglogprior(self, theta):
        """ p(m, s) ~ 1/s for uninformative """
        return -math.log(theta[0])

