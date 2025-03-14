import re
import math
import warnings
import numpy as np

from scipy.stats import genextreme, pearson3, gumbel_r
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm, genpareto
from scipy.optimize import brentq, minimize
from scipy.special import gamma, gammaln, polygamma

# Distribution names
MARGINAL_NAMES = {
    "Gumbel": 1,
    "LogNormal": 2,
    "GEV": 3,
    "LogPearson3": 4,
    "Normal": 5,
    "GeneralizedPareto": 6,
    "GeneralizedLogistic": 7,
    "Gamma": 8
    }

EULER_CONSTANT = 0.577215664901532

PARAMETERS = ["locn", "logscale", "shape1"]

# Bounds
LOCN_LOWER = -1e10
LOCN_UPPER = 1e10

LOGSCALE_LOWER = -20
LOGSCALE_UPPER = 20

SHAPE1_LOWER = -2.0
SHAPE1_UPPER = 2.0

SHAPE1_PRIOR_LOC_DEFAULT = 0.

SHAPE1_PRIOR_SCALE_DEFAULT = 0.2
SHAPE1_PRIOR_SCALE_MIN = 1e-10
SHAPE1_PRIOR_SCALE_MAX = 1.


def _prepare(data):
    data = np.array(data)
    if data.ndim != 1:
        errmess = f"Expected 1dim array, got ndim={data.ndim}."
        raise ValueError(errmess)

    data[np.isnan(data)] = -2e10
    data_sorted = np.sort(data, axis=0)
    data_sorted[data_sorted < -1e10] = np.nan
    nval = (~np.isnan(data_sorted)).sum()

    if nval < 4:
        errmess = f"Expected length of valid data>=4, got {nval}."
        raise ValueError(errmess)

    return data_sorted, nval


def _comb(n, i):
    """ Function much faster than scipy.comb """
    n = float(n)  # To avoid overflow if i == 8
    if i > n:
        return 0
    elif i == 0:
        return 1
    elif i == 1:
        return n
    elif i == 2:
        return n * (n - 1) / 2
    elif i == 3:
        return n * (n - 1) * (n - 2) / 6
    elif i == 4:
        return n*(n-1)*(n-2)*(n-3)/24
    elif i == 5:
        return n*(n-1)*(n-2)*(n-3)*(n-4)/120
    elif i == 6:
        return n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)/720
    elif i == 7:
        return n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)*(n-6)/5040
    elif i == 8:
        return n*(n-1)*(n-2)*(n-3)*(n-4)*(n-5)*(n-6)*(n-7)/40320


def _prepare_censored_data(data, low_censor):
    data = np.array(data)
    low_censor = -1e10 if low_censor is None else float(low_censor)
    dcens = data[data > low_censor]
    if len(dcens) < 5:
        errmess = "Expected at least 5 uncensored values"
        raise ValueError(errmess)
    return dcens, len(data)-len(dcens)


def _check_param_value(x, lower=-np.inf, upper=np.inf, name=None):
    name = "" if name is None else f"{name} "
    errmess = f"Invalid value for parameter {name}'{x}'."
    if x is None:
        raise ValueError(errmess)

    try:
        x = float(x)
    except Exception:
        raise ValueError(errmess)

    if np.isnan(x) or not np.isfinite(x):
        raise ValueError(errmess)

    if x < lower or x > upper:
        errmess += f" Expected value in [{lower}, {upper}]."
        raise ValueError(errmess)

    return x


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
    data_sorted, nval = _prepare(data)
    eta = int(eta)

    # Compute L moments, see ARR, book 3 page 47
    lam1, lam2, lam3, lam4 = 0, 0, 0, 0
    v1, v2, v3, v4 = 0, 0, 0, 0

    for i in range(1, nval+1):
        # Compute C factors (save time by reusing data)
        Cim1e0 = _comb(i-1, eta)
        Cim1e1 = _comb(i-1, eta+1)
        Cim1e2 = _comb(i-1, eta+2)

        Cnmi1 = nval-i
        Cnmi2 = _comb(nval-i, 2)
        Cnmi3 = _comb(nval-i, 3)

        # Compute components of moments
        d = data_sorted[i-1]
        v1 += Cim1e0*d
        v2 += (Cim1e1-Cim1e0*Cnmi1)*d
        v3 += (Cim1e2-2*Cim1e1*Cnmi1+Cim1e0*Cnmi2)*d

        if compute_lam4:
            v4 += (_comb(i - 1, eta + 3) - 3 * Cim1e2 * Cnmi1
                   + 3 * Cim1e1 * Cnmi2 - Cim1e0 * Cnmi3) * d

    lam1 = v1/_comb(nval, eta+1)
    lam2 = v2/_comb(nval, eta+2)/2
    lam3 = v3/_comb(nval, eta+3)/3
    if compute_lam4:
        lam4 = v4/_comb(nval, eta+4)/4
    else:
        lam4 = np.nan

    return lam1, lam2, lam3, lam4


def factory(distname):
    txt = "/".join(MARGINAL_NAMES.keys())
    if distname not in MARGINAL_NAMES:
        errmsg = f"Expected distnames in {txt}, got {distname}."
        raise ValueError(errmsg)

    if distname == "GEV":
        return GEV()
    elif distname == "LogPearson3":
        return LogPearson3()
    elif distname == "Gumbel":
        return Gumbel()
    elif distname == "LogNormal":
        return LogNormal()
    elif distname == "Normal":
        return Normal()
    elif distname == "GeneralizedPareto":
        return GeneralizedPareto()
    elif distname == "GeneralizedLogistic":
        return GeneralizedLogistic()
    elif distname == "Gamma":
        return Gamma()
    else:
        raise ValueError(errmsg)


class FloodFreqDistribution():
    """ Base class for flood frequency distribution """

    def __init__(self, name, has_shape=True):
        self.name = name

        self.has_shape = has_shape

        # bounds
        self.locn_lower = LOCN_LOWER
        self.locn_upper = LOCN_UPPER
        self.logscale_lower = LOGSCALE_LOWER
        self.logscale_upper = LOGSCALE_UPPER
        self.shape1_lower = SHAPE1_LOWER
        self.shape1_upper = SHAPE1_UPPER

        # Default values
        self._locn = np.nan
        self._logscale = np.nan

        # Priors
        self._shape1_prior_loc = SHAPE1_PRIOR_LOC_DEFAULT
        self._shape1_prior_scale = SHAPE1_PRIOR_SCALE_DEFAULT

    def __str__(self):
        txt = f"{self.name} flood frequency distribution:\n"
        txt += f"{' '*2}locn     = {self.locn:0.2f}\n"
        txt += f"{' '*2}logscale = {self.logscale:0.2f}\n"
        txt += f"{' '*2}shape1   = {self.shape1:0.2f}\n"
        try:
            x0, x1 = self.support
            txt += f"\n{' '*2}Support   = [{x0:0.3f}, {x1:0.3f}]\n"
        except NotImplementedError:
            txt += "\n"
        return txt

    def __setitem__(self, key, value):
        if key not in PARAMETERS:
            txt = "/".join(PARAMETERS)
            raise ValueError(f"Expected {key} in {txt}.")
        setattr(self, key, value)

    def __getitem__(self, key):
        if key not in PARAMETERS:
            txt = "/".join(list(self.params.names))
            raise ValueError(f"Expected {key} in {txt}.")
        return getattr(self, key)

    @property
    def locn(self):
        return self._locn

    @locn.setter
    def locn(self, value):
        self._locn = _check_param_value(value,
                                        self.locn_lower,
                                        self.locn_upper,
                                        "locn")

    @property
    def logscale(self):
        return self._logscale

    @logscale.setter
    def logscale(self, value):
        self._logscale = _check_param_value(value,
                                            self.logscale_lower,
                                            self.logscale_upper,
                                            "logscale")

    @property
    def scale(self):
        return math.exp(self.logscale)

    @property
    def shape1(self):
        if self.has_shape:
            return self._shape1
        else:
            return 0.

    @shape1.setter
    def shape1(self, value):
        if not self.has_shape:
            errmess = f"Try to set shape for distribution {self.name}."
            raise ValueError(errmess)

        self._shape1 = _check_param_value(value,
                                          self.shape1_lower,
                                          self.shape1_upper,
                                          "shape1")

    @property
    def params(self):
        return np.array([self.locn, self.logscale,
                         self.shape1])

    @params.setter
    def params(self, value):
        if len(value) != 3:
            errmess = "Expected 3 parameters, got {len(theta)}."
            raise ValueError(errmess)

        locn, logscale, shape1 = value
        self.locn = locn
        self.logscale = logscale
        if self.has_shape:
            self.shape1 = shape1

    @property
    def support(self):
        errmsg = f"Property support not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    @property
    def shape1_prior_loc(self):
        return self._shape1_prior_loc

    @shape1_prior_loc.setter
    def shape1_prior_loc(self, value):
        if not self.has_shape:
            errmess = "Try to set shape prior loc for"\
                      + f"distribution {self.name}."
            raise ValueError(errmess)

        self._shape1_prior_loc = _check_param_value(value,
                                                    SHAPE1_LOWER,
                                                    SHAPE1_UPPER,
                                                    "shape1_prior_loc")

    @property
    def shape1_prior_scale(self):
        return self._shape1_prior_scale

    @shape1_prior_scale.setter
    def shape1_prior_scale(self, value):
        if not self.has_shape:
            errmess = "Try to set shape prior loc for"\
                      + f"distribution {self.name}."
            raise ValueError(errmess)

        smin = SHAPE1_PRIOR_SCALE_MIN
        smax = SHAPE1_PRIOR_SCALE_MAX
        self._shape1_prior_scale = _check_param_value(value,
                                                      smin, smax,
                                                      "shape1_prior_scale")

    def in_support(self, x):
        x0, x1 = self.support
        ok = np.ones_like(x).astype(bool)
        return np.where((x >= x0) & (x <= x1), ok, ~ok)

    def params_guess(self, data):
        errmsg = "Method params_guess not implemented"\
                 + f" for class {self.name}."
        raise NotImplementedError(errmsg)

    def get_scipy_params(self):
        errmsg = "Method get_scipy_params not implemented"\
                 + f" for class {self.name}."
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

    def rvs(self, size):
        errmsg = f"Method rvs not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def neglogpost(self, theta, data_censored, low_censor, ncens):
        try:
            self.locn = theta[0]
            self.logscale = theta[1]
            if self.has_shape:
                self.shape1 = theta[2]
        except ValueError:
            return np.inf

        nlp = -self.logpdf(data_censored).sum()
        if ncens > 0:
            nlp -= self.logcdf(low_censor)*ncens

        if not np.isfinite(nlp) or np.isnan(nlp):
            return np.inf

        # Prior on shape param
        if self.has_shape:
            nlp -= norm.logpdf(theta[-1],
                               loc=self.shape1_prior_loc,
                               scale=self.shape1_prior_scale)
        return nlp

    def maximum_posterior_estimate(self, data, low_censor=None,
                                   nexplore=5000,
                                   explore_scale=0.2):
        # Prepare data
        dcens, ncens = _prepare_censored_data(data, low_censor)

        # Initial parameter exploration
        # random perturb guesses parameters in
        # arcsinh space (log for large values and lin for small values)
        self.params_guess(data)
        theta0 = self.params
        perturb = np.random.normal(loc=0, scale=explore_scale,
                                   size=(nexplore, len(theta0)))
        explore = np.sinh(np.arcsinh(theta0)[None, :] + perturb)
        # .. keep guessed parameter last
        explore[-1] = theta0

        # .. loop throught explored parameter and check loglike
        nlp_min = np.inf
        for theta in explore:
            nlp = self.neglogpost(theta, dcens, low_censor, ncens)
            if nlp < nlp_min and np.isfinite(nlp) and not np.isnan(nlp):
                theta0 = theta
                nlp_min = nlp

        # Nelder-Mead fit
        options = dict(xatol=1e-5, fatol=1e-5,
                       maxiter=10000, maxfev=50000)
        theta0 = theta0[:2] if not self.has_shape else theta0
        opt = minimize(self.neglogpost, theta0,
                       args=(dcens, low_censor, ncens),
                       method="Nelder-Mead",
                       options=options)
        self.locn = opt.x[0]
        self.logscale = opt.x[1]
        if self.has_shape:
            self.shape1 = opt.x[2]

        if not opt.success:
            warnmess = "Nelder-Mead optimisation did not converge."\
                       + f"Message: {opt.message}."
            warnings.warn(warnmess)

        return -opt.fun, self.params, dcens, ncens


class Normal(FloodFreqDistribution):
    def __init__(self):
        super(Normal, self).__init__("Normal", False)

    def get_scipy_params(self):
        return {"loc": self.locn, "scale": self.scale}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(norm, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return norm.rvs(size=size, **kw)
            return fun

        return super(Normal, self).__getattribute__(name)

    @property
    def support(self):
        return -np.inf, np.inf

    def params_guess(self, data):
        self.locn = np.mean(data)
        scale2 = ((data - self.locn)**2).mean()
        self.logscale = math.log(math.sqrt(scale2))

    def fit_lh_moments(self, data, eta=0):
        if eta != 0:
            errmsg = f"Expected eta=0, got {eta}."
            raise ValueError(errmsg)

        lam1, lam2, _, _ = lh_moments(data, eta, compute_lam4=False)
        sqpi = math.sqrt(math.pi)
        self.locn = lam1
        self.logscale = math.log(lam2*sqpi)


class GEV(FloodFreqDistribution):
    """ GEV distribution class"""

    def __init__(self):
        super(GEV, self).__init__("GEV")

    @property
    def support(self):
        x0 = self.locn+self.scale/self.shape1
        if self.shape1 < 0:
            return x0, np.inf
        else:
            return -np.inf, x0

    def get_scipy_params(self):
        return {"c": self.shape1, "loc": self.locn, "scale": self.scale}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(genextreme, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return genextreme.rvs(size=size, **kw)
            return fun

        return super(GEV, self).__getattribute__(name)

    def params_guess(self, data):
        try:
            # Try LH moments with decrasing eta to
            # favour high eta if possible.
            for eta in [2, 1, 0]:
                self.fit_lh_moments(data, eta)

                # Check support is ok to avoid
                # problems with likelihood computation later on
                if np.all(self.in_support(data)):
                    break

            if not np.all(self.in_support(data)):
                raise ValueError()

        except Exception:
            # Revert to moment matching of Gumbel distribution
            # See https://en.wikipedia.org/wiki/Gumbel_distribution
            alpha = data.var()*6/math.pi**2
            self.logscale = math.log(alpha)
            self.locn = data.mean()-EULER_CONSTANT*alpha
            self.shape1 = 0.01

    def fit_lh_moments(self, data, eta=0):
        """ See Wang et al. (1997). """
        if eta < 0 or eta > 3:
            errmsg = f"Expected eta in [0, 3], got {eta}."
            raise ValueError(errmsg)

        # Get LH moments
        lam1, lam2, lam3, _ = lh_moments(data, eta, compute_lam4=False)
        tau3 = lam3/lam2

        # See Wang et al.(1997), Equation 26
        coefs_all = {
            0: [0.2849, -1.8213, 0.8140, -0.2835],
            1: [0.4823, -2.1494, 0.7269, -0.2103],
            2: [0.5914, -2.3351, 0.6442, -0.1616],
            3: [0.6618, -2.4548, 0.5733, -0.1273],
            4: [0.7113, -2.5383, 0.5142, -0.1027]
        }
        coefs = coefs_all[eta]
        kappa_ini = coefs[0]+coefs[1]*tau3+coefs[2]*tau3**2+coefs[3]*tau3**3
        # .. avoids large errors of interpolation
        kappa_ini = 0.01 if abs(kappa_ini) > 1 else kappa_ini

        # Solve equation because kappa is outside of validity limits
        # See Wang et al (1997), Ratio between Eq. 21 and Eq. 20
        def fun(k):
            A = (eta+3)/(eta+2)/3
            B = -(eta+4)*(eta+3)**(-k)
            C = 2*(eta+3)*(eta+2)**(-k)-(eta+2)*(eta+1)**(-k)
            D = (-(eta+2)**(-k)+(eta+1)**(-k))
            return A*(B+C)/D-tau3

        # .. finds proper bounds
        deltak = 0.1
        k0 = kappa_ini-deltak
        niter = 0
        while fun(k0) < 0 and niter < 10:
            k0 -= deltak
            niter += 1

        k1 = kappa_ini+deltak
        niter = 0
        while fun(k1) > 0 and niter < 10:
            k1 += deltak
            niter += 1

        # .. solve within bounds
        f0, f1 = fun(k0), fun(k1)
        if f0*f1 < 0:
            kappa = brentq(fun, k0, k1)
        else:
            # .. failed getting proper bounds
            kappa = kappa_ini

        # .. reasonable bounds on kappa
        # .. cannot have kappa<-1
        kappa = max(-0.95, min(SHAPE1_UPPER, kappa))

        # See wang et al. (1997), Equation 19
        g = gamma(1+kappa)
        e = (1+eta)**(-kappa)
        lhs = (eta+2)*g/2/kappa*(-(eta+2)**(-kappa)+e)
        alpha = lam2/lhs

        # See wang et al. (1997), Equation 18
        lhs = alpha/kappa*(1-g*e)
        tau = lam1-lhs

        self.shape1 = kappa
        self.logscale = math.log(alpha)
        self.locn = tau


class LogPearson3(FloodFreqDistribution):
    """ Log Pearson III distribution class"""

    def __init__(self):
        super(LogPearson3, self).__init__("LogPearson3")

    def get_scipy_params(self):
        return {"skew": self.shape1, "loc": self.locn, "scale": self.scale}

    @property
    def alpha(self):
        return 4 / self.shape1**2

    @property
    def beta(self):
        return np.sign(self.shape1) * math.sqrt(self.alpha) / self.scale

    @property
    def tau(self):
        return self.locn - self.alpha / self.beta

    @property
    def support(self):
        if self.beta < 0:
            return [0, math.exp(max(-100, min(100, self.tau)))]
        else:
            return [math.exp(max(-100, min(100, self.tau))), np.inf]

    def pdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.pdf(np.log(x), **kw) / x

    def logpdf(self, x):
        kw = self.get_scipy_params()
        lx = np.log(x)
        return pearson3.logpdf(lx, **kw) - lx

    def cdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.cdf(np.log(x), **kw)

    def logcdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.logcdf(np.log(x), **kw)

    def ppf(self, q):
        kw = self.get_scipy_params()
        return np.exp(pearson3.ppf(q, **kw))

    def params_guess(self, data):
        try:
            self.fit_lh_moments(data)
            if not np.all(self.in_support(data)):
                raise ValueError()

        except Exception:
            # Problem with gam, revert back to log norm
            self.shape1 = 0.01
            logx = np.log(data[data > 0])
            self.locn = logx.mean()
            self.logscale = math.log((logx-self.locn).std(ddof=1))

        # Use guess param value as prior loc for shape
        self.shape1_prior_loc = self.shape1

    def rvs(self, size):
        kw = self.get_scipy_params()
        return np.exp(pearson3.rvs(size=size, **kw))

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking (1990), http://lib.stat.cmu.edu/general/lmoments """
        if eta > 0:
            errmess = f"Expected eta=0, got {eta}."
            raise ValueError(errmess)

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
        C1, C2, C3 = 0.2906, 0.1882, 0.0442
        D1, D2, D3 = 0.36067, -0.59567, 0.25361
        D4, D5, D6 = -2.78861, 2.56096, -0.77045
        pi3, rootpi = 9.4247780, 1.7724539
        SMALL = 1e-15
        # XMOM(1) -> lam1
        # XMOM(2) -> lam2
        # XMOM(3) -> tau3

        T3 = np.abs(tau3)

        alpha = None
        if lam2 <= 0 or T3 >= 1:
            mu = 0.
            sigma = 0.
            gam = 0.
        elif T3 <= SMALL:
            mu = lam1
            sigma = lam2*rootpi
            gam = 0.
        elif T3 >= 1./3:
            alpha = 0.
            Ti3 = 1.-T3
            alpha = Ti3*(D1+Ti3*(D2+Ti3*D3))/(1.+Ti3*(D4+Ti3*(D5+Ti3*D6)))
        else:
            Ti4 = pi3*T3*T3
            alpha = (1.+C1*Ti4)/(Ti4*(1.+Ti4*(C2+Ti4*C3)))

        if alpha is not None:
            rtalpha = math.sqrt(alpha)

            # .. check reasonable bounds
            if 2./rtalpha < SHAPE1_LOWER:
                rtalpha = 2./SHAPE1_LOWER
                alpha = rtalpha**2
            elif 2./rtalpha > SHAPE1_UPPER:
                rtalpha = 2./SHAPE1_UPPER
                alpha = rtalpha**2

            beta = rootpi*lam2*math.exp(gammaln(alpha)-gammaln(alpha+0.5))
            mu = lam1
            sigma = beta*rtalpha
            gam = 2./rtalpha
            if tau3 < 0:
                gam *= -1

        self.shape1 = gam
        self.logscale = math.log(sigma)
        self.locn = mu


class Gumbel(FloodFreqDistribution):
    """ GEV distribution class"""

    def __init__(self):
        super(Gumbel, self).__init__("Gumbel", False)

    @property
    def support(self):
        return -np.inf, np.inf

    def get_scipy_params(self):
        return {"loc": self.locn, "scale": self.scale}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(gumbel_r, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                return gumbel_r.rvs(size=size, scale=self.scale, loc=self.locn)
            return fun

        return super(Gumbel, self).__getattribute__(name)

    def fit_lh_moments(self, data, eta=0):
        """ See Wang et al. (1997). """
        if eta < 0 or eta > 4:
            errmsg = f"Expected eta in [0, 4], got {eta}."
            raise ValueError(errmsg)

        # Get LH moments
        lam1, lam2, lam3, _ = lh_moments(data, eta, compute_lam4=False)

        # See wang et al. (1997), Equation 23
        lhs = (eta+2)/2*(math.log(eta+2)-math.log(eta+1))
        alpha = lam2/lhs

        # See wang et al. (1997), Equation 22
        lhs = alpha*(EULER_CONSTANT+math.log(eta+1))
        tau = lam1-lhs

        self.logscale = math.log(alpha)
        self.locn = tau

    def params_guess(self, data):
        try:
            # Try LH moments with decrasing eta to
            # favour high eta if possible.
            for eta in [2, 1, 0]:
                self.fit_lh_moments(data, eta)

                # Check support is ok to avoid
                # problems with likelihood computation later on
                if np.all(self.in_support(data)):
                    break

            if not np.all(self.in_support(data)):
                raise ValueError()

        except Exception:
            # Revert to moment matching
            # See https://en.wikipedia.org/wiki/Gumbel_distribution
            alpha = data.var()*6/math.pi**2
            self.logscale = math.log(alpha)
            self.locn = data.mean()-EULER_CONSTANT*alpha
            self.shape1 = 0.01


class LogNormal(FloodFreqDistribution):
    """ LogNormal distribution class"""

    def __init__(self):
        super(LogNormal, self).__init__("LogNormal", False)

        # locn in log space
        self.locn_lower = -1e2
        self.locn_upper = 1e2

    @property
    def support(self):
        return 0, np.inf

    def get_scipy_params(self):
        return {"s": self.scale, "loc": 0., "scale": math.exp(self.locn)}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(lognorm, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return lognorm.rvs(size=size, **kw)
            return fun

        return super(LogNormal, self).__getattribute__(name)

    def params_guess(self, data):
        data_ok = data[~np.isnan(data) & (data > 0)]
        if len(data_ok) <= 3:
            errmess = "Expected at least 3 samples valid and  > 0."
            raise ValueError(errmess)

        # Maximum likelihood
        logx = np.log(data_ok)
        self.locn = logx.mean()
        scale2 = ((logx-self.locn)**2).mean()
        self.logscale = math.log(math.sqrt(scale2))

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking and Wallis (1997), Appendix, page 198. """
        if eta != 0:
            errmsg = f"Expected eta=0, got {eta}."
            raise ValueError(errmsg)

        # Get LH moments
        lx = np.log(data[data > 0])
        lam1, lam2, _, _ = lh_moments(lx, eta, compute_lam4=False)

        self.locn = lam1
        self.logscale = math.log(lam2*math.sqrt(math.pi))


class GeneralizedPareto(FloodFreqDistribution):
    """ Generalized Pareto distribution class"""

    def __init__(self):
        super(GeneralizedPareto, self).__init__("GeneralizedPareto")

    @property
    def support(self):
        tau, alpha, kappa = self.locn, self.scale, self.shape1
        if kappa < 0:
            return tau, np.inf
        else:
            return tau, tau+alpha/kappa

    def get_scipy_params(self):
        return {"c": -self.shape1, "loc": self.locn, "scale": self.scale}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(genpareto, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return genpareto.rvs(size=size, **kw)
            return fun

        return super(GeneralizedPareto, self).__getattribute__(name)

    def params_guess(self, data):
        self.fit_lh_moments(data)

        # .. Correct if lh moments is failing
        smin, smax = self.support
        dmin, dmax = data.min(), data.max()
        tau, alpha = self.locn, self.scale
        if smin > dmin:
            self.locn = dmin-1e-3

        smin, smax = self.support
        tau, alpha, kappa = self.locn, self.scale, self.shape1
        if smax < dmax and kappa > 0:
            self.shape1 = alpha/(dmax-tau)*0.99

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking and Wallis (1997), Appendix, page 195. """
        if eta != 0:
            errmsg = f"Expected eta=0, got {eta}."
            raise ValueError(errmsg)

        # Get L moments
        lam1, lam2, lam3, _ = lh_moments(data, eta, compute_lam4=False)
        tau3 = lam3/lam2

        kappa = (1-3*tau3)/(1+tau3)
        self.shape1 = kappa
        self.locn = lam1-(2+kappa)*lam2
        self.logscale = math.log((1+kappa)*(2+kappa)*lam2)


class GeneralizedLogistic(FloodFreqDistribution):
    """ Generalized Logistic distribution class"""

    def __init__(self):
        super(GeneralizedLogistic, self).__init__("GeneralizedLogistic")
        self.kappa_transition = 1e-10

    @property
    def support(self):
        tau, alpha, kappa = self.locn, self.scale, self.shape1
        kt = self.kappa_transition
        if kappa < -kt:
            return tau+alpha/kappa, np.inf
        elif kappa > kt:
            return -np.inf, tau+alpha/kappa
        else:
            return -np.inf, np.inf

    def get_scipy_params(self):
        return {}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf"]:
            kt = self.kappa_transition
            tau, alpha, kappa = self.locn, self.scale, self.shape1

            def fun(x):
                z = (x-tau)/alpha
                if abs(kappa) > kt:
                    u = 1 - kappa * z
                    z = np.where(u > 1e-100, -1.0/kappa*np.log(u), np.nan)

                if name == "pdf":
                    return 1./alpha*np.exp(-(1-kappa)*z)/(1+np.exp(-z))**2
                else:
                    return 1./(1+np.exp(-z))

            return fun

        elif name in ["logpdf", "logcdf"]:
            f = getattr(self, re.sub("log", "", name))

            def fun(x):
                return np.log(f(x))

            return fun

        elif name == "ppf":
            kt = self.kappa_transition
            tau, alpha, kappa = self.locn, self.scale, self.shape1

            def fun(p):
                if abs(kappa) > kt:
                    return tau+alpha/kappa*(1-(1.0/p-1)**kappa)
                else:
                    return tau+alpha*np.log(1.0/p-1)

            return fun

        elif name == "rvs":
            def fun(size):
                u = np.random.uniform(0, 1, size=size)
                return self.ppf(u)
            return fun

        return super(GeneralizedLogistic, self).__getattribute__(name)

    def params_guess(self, data):
        self.fit_lh_moments(data)

        # .. Correct if lh moments is failing
        smin, smax = self.support
        dmin, dmax = data.min(), data.max()
        tau, alpha = self.locn, self.scale
        if smin > dmin:
            self.shape1 = alpha/(dmin-tau)*0.99
        elif smax < dmax:
            self.shape1 = alpha/(dmax-tau)*0.99

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking and Wallis (1997), Appendix, page 197. """
        if eta != 0:
            errmsg = f"Expected eta=0, got {eta}."
            raise ValueError(errmsg)

        # Get L moments
        lam1, lam2, lam3, _ = lh_moments(data, eta, compute_lam4=False)
        tau3 = lam3/lam2

        kappa = -tau3
        alpha = lam2*math.sin(kappa*math.pi)/kappa/math.pi
        tau = lam1-alpha*(1/kappa-math.pi/math.sin(kappa*math.pi))

        self.shape1 = kappa
        self.logscale = math.log(alpha)
        self.locn = tau


class Gamma(FloodFreqDistribution):
    """ Gamma distribution class"""

    def __init__(self):
        super(Gamma, self).__init__("Gamma", False)

        # Only positive locn param allowed
        self.locn_lower = 1e-10

    @property
    def support(self):
        return 0, np.inf

    def get_scipy_params(self):
        loc, scale = self.locn, self.scale
        return {"a": loc/scale, "scale": scale}

    def __getattribute__(self, name):
        if name in ["pdf", "cdf", "ppf", "logpdf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(gamma_dist, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return gamma_dist.rvs(size=size, **kw)
            return fun

        return super(Gamma, self).__getattribute__(name)

    def params_guess(self, data):
        data_ok = data[~np.isnan(data) & (data > 0)]
        if len(data_ok) <= 5:
            errmess = "Expected at least 5 samples valid and  > 0."
            raise ValueError(errmess)

        # See https://en.wikipedia.org/wiki/Gamma_distribution
        #              #Maximum_likelihood_estimation
        dm = data_ok.mean()
        dlm = np.log(data_ok).mean()
        s = math.log(dm) - dlm
        a = (3 - s + math.sqrt((s - 3)**2 + 24*s)) / 12 / s
        nit = 0
        while True and nit < 100:
            up = (math.log(a) - polygamma(0, a) - s)/(1. / a - polygamma(1, a))
            anew = a - up
            if abs(anew - a) < 1e-10:
                break
            a = anew
            nit += 1

        self.locn = dm
        self.logscale = math.log(dm/a)
