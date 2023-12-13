import sys
import math, re
import numpy as np
import pandas as pd

from scipy.stats import poisson, nbinom, bernoulli

from tqdm import tqdm

from hydrodiy.data.containers import Vector

# Distribution names
DISCRETE_NAMES = {
    "Poisson": 1, \
    "NegativeBinomial": 2, \
    "Bernoulli": 3
}

PARAMETERS = ["locn", "phi"]

# Bounds on discrete parameters
PHI_LOWER = 0.1
PHI_UPPER = 100.0

LOCN_LOWER = 0
LOCN_UPPER = 100

# Bounds on discrete data
NEVENT_UPPER = 100


def factory(distname):
    txt = "/".join(DISCRETE_NAMES.keys())
    errmsg = f"Expected distnames in {txt}, got {distname}."
    assert distname in DISCRETE_NAMES, errmsg

    if distname == "Poisson":
        return Poisson()
    elif distname == "NegativeBinomial":
        return NegativeBinomial()
    elif distname == "Bernoulli":
        return Bernoulli()
    else:
        raise ValueError(errmsg)


class DiscreteDistribution():
    """ Base class for flood frequency distribution """

    def __init__(self, name):
        self.name = name
        self.isbern = name == "Bernoulli"
        self._locn = np.nan
        self._phi = np.nan

    def __str__(self):
        txt = f"{self.name} flood frequency distribution:\n"
        txt += f"{' '*2}locn = {self.locn:0.2f}\n"
        txt += f"{' '*2}phi  = {self.phi:0.2f}\n"
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
        if np.isnan(self._locn):
            raise ValueError("locn is not set.")
        return self._locn

    @locn.setter
    def locn(self, value):
        locn = float(value)
        locn_upper = 1 if self.isbern else LOCN_UPPER
        assert locn>=LOCN_LOWER and locn<=locn_upper, \
                f"Expected locn in ]{LOCN_LOWER}, {locn_upper}[, "+\
                f"got {locn}."
        self._locn = locn

    @property
    def phi(self):
        if np.isnan(self._phi):
            raise ValueError("phi is not set.")
        return self._phi

    @phi.setter
    def phi(self, value):
        phi = float(value)
        assert phi>=PHI_LOWER and phi<=PHI_UPPER, \
                f"Expected phi in ]{PHI_LOWER}, {PHI_UPPER}[, "+\
                f"got {phi}."
        self._phi = phi

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

    def logpmf(self, x):
        errmsg = f"Method logpmf not implemented for class {self.name}."
        raise NotImplementedError(errmsg)

    def pmf(self, x):
        errmsg = f"Method pmf not implemented for class {self.name}."
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


    def pot2ams_cdf(self, pot_cdf, kmax=100):
        """ Convert pot cdf to ams cdf """
        if self.isbern:
            raise ValueError("Function not implemented for Bernoulli variable.")

        pot_cdf = np.atleast_1d(pot_cdf)
        assert pot_cdf.ndim==1, "Expected pot_cdf as 1D array."
        pot_cdf = pot_cdf[:, None]

        if self.name == "Poisson":
            ams_cdf = np.exp(-self.locn*(1-pot_cdf))
        else:
            kk = np.arange(kmax)
            pp = self.pmf(kk)[None, :]
            ams_cdf = np.sum(pp*(pot_cdf**kk[None, :]), axis=1)

        ams_cdf = ams_cdf.squeeze()
        notcool = (ams_cdf<0) | (ams_cdf>1) | np.iscomplex(ams_cdf)
        ams_cdf[notcool] = np.nan

        return ams_cdf


    def ams2pot_cdf(self, ams_cdf, kmax=100):
        """ Convert ams cdf to pot cdf """
        if self.isbern:
            raise ValueError("Function not implemented for Bernoulli variable.")

        ams_cdf = np.atleast_1d(ams_cdf)
        assert ams_cdf.ndim==1, "Expected ams_cdf as 1D array."
        pot_cdf = np.zeros_like(ams_cdf)

        if self.name == "Poisson":
            pot_cdf = 1+np.log(ams_cdf)/self.locn
        else:
            kk = np.arange(kmax)
            coefs0 = self.pmf(kk)
            for i, ac in enumerate(ams_cdf):
                if np.isnan(ac) or np.isinf(ac):
                    pot_cdf[i] = ac
                    continue

                coefs = coefs0.copy()
                coefs[0] -= ac
                roots = np.polynomial.Polynomial(coefs).roots()

                roots = roots[(roots.real>0)]
                roots = roots[np.argmin(np.abs(roots.imag))]
                pot_cdf[i] = roots.real

        pot_cdf = pot_cdf.squeeze()
        notcool = (pot_cdf<0) | (pot_cdf>1) | np.iscomplex(pot_cdf)
        pot_cdf[notcool] = np.nan

        return pot_cdf



class Poisson(DiscreteDistribution):
    def __init__(self):
        super(Poisson, self).__init__("Poisson")

    def get_scipy_params(self):
        return {"mu": self.locn}

    def __getattribute__(self, name):
        if name in ["pmf", "cdf", "ppf", "logpmf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(poisson, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return poisson.rvs(size=size, **kw)
            return fun

        return super(Poisson, self).__getattribute__(name)

    @property
    def support(self):
        return 0, sys.maxsize-1

    def params_guess(self, data):
        self.locn = np.mean(data)



class NegativeBinomial(DiscreteDistribution):
    def __init__(self):
        super(NegativeBinomial, self).__init__("NegativeBinomial")

    def get_scipy_params(self):
        klocn, kphi = self.locn, self.phi
        v = klocn+klocn**2/kphi
        n = klocn**2/(v-klocn)
        p = klocn/v
        return {"n": n, "p": p}


    def __getattribute__(self, name):
        if name in ["pmf", "cdf", "ppf", "logpmf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(nbinom, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return nbinom.rvs(size=size, **kw)
            return fun

        return super(NegativeBinomial, self).__getattribute__(name)

    @property
    def support(self):
        return 0, sys.maxsize-1

    def params_guess(self, data):
        self.locn = np.mean(data)
        self.phi = 1.



class Bernoulli(DiscreteDistribution):
    def __init__(self):
        super(Bernoulli, self).__init__("Bernoulli")

    def get_scipy_params(self):
        return {"p": self.locn}


    def __getattribute__(self, name):
        if name in ["pmf", "cdf", "ppf", "logpmf", "logcdf"]:
            def fun(x):
                kw = self.get_scipy_params()
                f = getattr(bernoulli, name)
                return f(x, **kw)
            return fun
        elif name == "rvs":
            def fun(size):
                kw = self.get_scipy_params()
                return bernoulli.rvs(size=size, **kw)
            return fun

        return super(Bernoulli, self).__getattribute__(name)

    @property
    def support(self):
        return 0, sys.maxsize-1

    @property
    def locn(self):
        return self._locn

    @locn.setter
    def locn(self, value):
        locn = float(value)
        assert locn>=LOCN_LOWER and locn<=1, \
                f"Expected locn in ]{LOCN_LOWER}, 1[, "+\
                f"got {locn}."
        self._locn = locn


    def params_guess(self, data):
        self.locn = np.mean(data)

