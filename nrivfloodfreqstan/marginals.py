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

DISTRIBUTION_NAMES = ["Normal", "GEV", "LogPearson3", \
                        "Gumbel", "LogNormal"]

EULER_CONSTANT = 0.577215664901532

PARAMETERS = ["locn", "logscale", "shape1"]

SHAPE1_MIN = -3
SHAPE1_MAX = 3

def _prepare(data):
    data = np.array(data)
    assert data.ndim == 1, f"Expected 1dim array, got ndim={data.ndim}."

    data[np.isnan(data)] = -2e100
    data_sorted = np.sort(data, axis=0)
    data_sorted[data_sorted<-1e100] = np.nan
    nval = (~np.isnan(data_sorted)).sum()

    errmsg = f"Expected length of valid data>=4, got {nval}."
    assert nval>=4, errmsg
    return data_sorted, nval


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
            v4 += (_comb(i-1, eta+3)-3*Cim1e2*Cnmi1\
                            +3*Cim1e1*Cnmi2-Cim1e0*Cnmi3)*d

    lam1 = v1/_comb(nval, eta+1)
    lam2 = v2/_comb(nval, eta+2)/2
    lam3 = v3/_comb(nval, eta+3)/3
    if compute_lam4:
        lam4 = v4/_comb(nval, eta+4)/4
    else:
        lam4 = np.nan

    return lam1, lam2, lam3, lam4


def factory(distname):
    txt = "/".join(DISTRIBUTION_NAMES)
    errmsg = f"Expected distnames in {txt}, got {distname}."
    assert distname in DISTRIBUTION_NAMES, errmsg

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
    else:
        raise ValueError(errmsg)


class FloodFreqDistribution():
    """ Base class for flood frequency distribution """

    def __init__(self, name):
        self.name = name
        self._locn = np.nan
        self._logscale = np.nan
        self._shape1 = 0. # Assume 0 shape for distribution that do not have shape

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
        if not key in PARAMETERS:
            txt = "/".join(PARAMETERS)
            raise ValueError(f"Expected {key} in {txt}.")
        setattr(self, key, value)

    def __getitem__(self, key):
        if not key in PARAMETERS:
            txt = "/".join(list(self.params.names))
            raise ValueError(f"Expected {key} in {txt}.")
        return getattr(self, key)

    @property
    def locn(self):
        return self._locn

    @locn.setter
    def locn(self, value):
        self._locn = float(value)

    @property
    def logscale(self):
        return self._logscale

    @logscale.setter
    def logscale(self, value):
        self._logscale = float(value)

    @property
    def scale(self):
        return math.exp(self.logscale)

    @property
    def shape1(self):
        return self._shape1

    @shape1.setter
    def shape1(self, value):
        value = float(value)
        assert value>=SHAPE1_MIN and value<=SHAPE1_MAX, \
                f"Expected shape in ]{SHAPE1_MIN}, {SHAPE1_MAX}[, "+\
                f"got {value}."
        self._shape1 = value

    @property
    def support(self):
        errmsg = f"Property support not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def in_support(self, x):
        x0, x1 = self.support
        ok = np.ones_like(x).astype(bool)
        return np.where((x>=x0)&(x<=x1), ok, ~ok)

    def get_scipy_params(self):
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

    def rvs(self, size):
        errmsg = f"Method rvs not implemented for class {self.name}."
        raise NotImplementedError(errmsg)


class Normal(FloodFreqDistribution):
    def __init__(self):
        super(Normal, self).__init__("Normal")

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
        self.logscale = math.log(np.std(data, ddof=1))

    def fit_lh_moments(self, data, eta=0):
        errmsg = f"Expected eta=0, got {eta}."
        assert eta==0, errmsg
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
        if self.shape1<0:
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
                self.fit_lh_moments(data)

                # Check support is ok to avoid
                # problems with likelihood computation later on
                assert np.all(self.in_support(data))
                break
        except:
            alpha = data.var()*6/math.pi**2
            self.logscale = math.log(alpha)
            self.locn = data.mean()-EULER_CONSTANT*alpha
            self.shape1 = 0.01

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
        # .. reasonable bounds on kappa
        kappa = min(SHAPE1_MAX, max(SHAPE1_MIN, kappa))

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
        return 4/self.shape1**2

    @property
    def beta(self):
        return np.sign(self.shape1)*math.sqrt(self.alpha)/self.scale

    @property
    def tau(self):
        return self.locn-self.alpha/self.beta

    @property
    def support(self):
        if self.beta<0:
            return [0, math.exp(max(-100, min(100, self.tau)))]
        else:
            return [math.exp(max(-100, min(100, self.tau))), np.inf]

    def pdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.pdf(np.log(x), **kw)/x

    def logpdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.logpdf(np.log(x), **kw)-np.log(x)

    def cdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.cdf(np.log(x), **kw)

    def logcdf(self, x):
        kw = self.get_scipy_params()
        return pearson3.logcdf(np.log(x), **kw)

    def ppf(self, q):
        kw = self.get_scipy_params()
        qq = q if self.shape1>0 else 1-q
        return np.exp(pearson3.ppf(qq, **kw))

    def params_guess(self, data):
        try:
            self.fit_lh_moments(data)
            assert np.all(self.in_support(data))
        except:
            logx = np.log(data)
            self.shape1 = max(SHAPE1_MIN, min(SHAPE1_MAX, skew(logx)))
            self.locn = logx.mean()
            self.logscale = math.log(logx.std(ddof=1))

    def rvs(self, size):
        kw = self.get_scipy_params()
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
        ALPHA = None
        if lam2<=0 or T3>=1:
            mu = 0.
            sigma = 0.
            gam = 0.

        elif T3<=SMALL:
            mu = lam1
            sigma = lam2*ROOTPI
            gam = 0.

        elif T3>=1./3:
            ALPHA = 0.
            Ti3 = 1.-T3
            ALPHA = Ti3*(D1+Ti3*(D2+Ti3*D3))/(1.+Ti3*(D4+Ti3*(D5+Ti3*D6)))

        else:
            Ti4 = PI3*T3*T3
            ALPHA = (1.+C1*Ti4)/(Ti4*(1.+Ti4*(C2+Ti4*C3)))

        if not ALPHA is None:
            RTALPH = math.sqrt(ALPHA)

            # .. check reasonable bounds
            if 2./RTALPH < SHAPE1_MIN:
                RTALPH = 2./SHAPE1_MIN
                ALPHA = RTALPHA**2
            elif 2./RTALPH > SHAPE1_MAX:
                RTALPH = 2./SHAPE1_MAX
                ALPHA = RTALPHA**2

            BETA = ROOTPI*lam2*math.exp(gammaln(ALPHA)-gammaln(ALPHA+0.5))
            mu = lam1
            sigma = BETA*RTALPH
            gam = 2./RTALPH
            if tau3<0:
                gam *= -1

        self.shape1 = gam
        self.logscale = math.log(sigma)
        self.locn = mu



class Gumbel(FloodFreqDistribution):
    """ GEV distribution class"""

    def __init__(self):
        super(Gumbel, self).__init__("Gumbel")

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

        self.logscale = math.log(alpha)
        self.locn = tau

    def params_guess(self, data):
        alpha = data.var()*6/math.pi**2
        self.logscale = math.log(alpha)
        self.locn = data.mean()-EULER_CONSTANT*alpha



class LogNormal(FloodFreqDistribution):
    """ LogNormal distribution class"""

    def __init__(self):
        super(LogNormal, self).__init__("LogNormal")

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
        logx = np.log(data)
        self.locn = logx.mean()
        self.logscale = math.log((logx-self.locn).std(ddof=1))

    def fit_lh_moments(self, data, eta=0):
        """ See Hosking and Wallis (1997), Appendix, page 198. """
        errmsg = f"Expected eta=0, got {eta}."
        assert eta==0, errmsg

        # Get LH moments
        lx = np.log(data)
        lam1, lam2, _, _ = lh_moments(lx, eta, compute_lam4=False)

        self.locn = lam1
        self.logscale = math.log(lam2*math.sqrt(math.pi))


