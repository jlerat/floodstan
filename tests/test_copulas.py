import json, re, math
from itertools import product as prod

import numpy as np
import pandas as pd

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.stats import ks_2samp
from scipy.integrate import quad, nquad

from statsmodels.distributions.copula.api import GaussianCopula, \
                                            ClaytonCopula, \
                                            GumbelCopula
import pytest
import warnings

from nrivfloodfreqstan import copulas

#from tqdm import tqdm


# ---------------- UTITILITIES -------------------------------------------

# Dummy copula to test interface
class DummyCopula(copulas.Copula):
    def __init__(self):
        super(DummyCopula, self).__init__("Dummy")

    def ppf_ucensored(self, ucensor, q):
        return q


# Wrapper around statsmodels copula
RHO_MIN = copulas.RHO_MIN
RHO_MAX = copulas.RHO_MAX

class SMCopula(copulas.Copula):
    def __init__(self, copula):
        super(SMCopula, self).__init__(f"SM-{copula}")
        self.copula = copula

    @copulas.Copula.rho.setter
    def rho(self, val):
        """ Set correlation parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_MIN}, {RHO_MAX}], got {rho}."
        assert rho>=RHO_MIN and rho<=RHO_MAX, errmsg
        self._rho = rho

        if self.copula == "Gaussian":
            self._smcop = GaussianCopula(rho)
        elif self.copula == "Gumbel":
            self._smcop = GumbelCopula(1/(1-rho))
        elif self.copula == "Clayton":
            self._smcop = ClaytonCopula(2/(1-rho))

    def pdf(self, uv):
        return self._smcop.pdf(uv)

    def cdf(self, uv):
        return self._smcop.cdf(uv)

    def sample(self, nsamples):
        return self._smcop.rvs(nsamples)


# ---------------- TESTS -------------------------------------------
def test_base_class():
    cop = DummyCopula()

    assert cop.name == "Dummy"

    # Test set/get of rho parameters
    msg = r"Rho \(nan\)"
    with pytest.raises(ValueError, match=msg):
        rho = cop.rho

    msg = f"Expected rho"
    with pytest.raises(AssertionError, match=msg):
        cop.rho = 100

    # test random sample
    nsamples = 10000
    uv = cop.sample(nsamples)
    assert uv.ndim == 2
    assert uv.shape[1] == 2
    assert uv.shape[0] == nsamples
    assert np.all(uv.min(axis=0)>0)
    assert np.all(uv.max(axis=0)<1)


def test_vs_statsmodels(allclose):
    ndata = 1000
    nsamples = 100000
    for copula in ["Gaussian", "Gumbel"]:
        print(f"\nTesting {copula}")
        cop1 = SMCopula(copula)
        cop2 = copulas.factory(copula)
        #cop2.approx = False
        uv = np.random.uniform(0, 1, size=(ndata, 2))

        for rho in np.linspace(0.05, 0.8, 10):
            cop1.rho = rho
            cop2.rho = rho

            pdf1 = cop1.pdf(uv)
            pdf2 = cop2.pdf(uv)
            assert allclose(pdf1, pdf2)

            cdf1 = cop1.cdf(uv)
            cdf2 = cop2.cdf(uv)
            assert allclose(cdf1, cdf2, atol=1e-5)

            uv1 = cop1.sample(nsamples)
            uv2 = cop2.sample(nsamples)
            if copula == "Gaussian":
                pq = norm.ppf(uv2)
                corr = np.corrcoef(pq.T)[0, 1]
                assert allclose(corr, rho, rtol=0, atol=1e-2)

            sa, pva = ks_2samp(uv1[:, 0], uv2[:, 0])
            sb, pvb = ks_2samp(uv1[:, 1], uv2[:, 1])
            assert pva>0.01 and pvb>0.01



def test_gaussian(allclose):
    cop = copulas.GaussianCopula()

    nsamples = 100
    uv = np.random.uniform(0, 1, size=(nsamples, 2))
    pq = norm.ppf(uv)
    mu = np.zeros(2)

    for rho in np.linspace(0, 0.8, 4):
        cop.rho = rho

        # Test pdf
        pdf = cop.pdf(uv)
        Sigma = np.array([[1, rho], [rho, 1]])
        expected = multivariate_normal.pdf(pq, mean=mu, cov=Sigma)
        assert allclose(pdf, expected)

        # Test cdf
        cdf = cop.cdf(uv)

        def fun(x, y):
            z = x*x-2*rho*x*y+y*y
            r2 = 1-rho*rho
            return math.exp(-z/r2/2)/2/math.pi/math.sqrt(r2)

        expected = np.zeros_like(cdf)
        for i in range(nsamples):
            c1, c2 = pq[i]
            expected[i], err = nquad(fun, [[-np.inf, c1], [-np.inf, c2]])

        assert allclose(cdf, expected, rtol=0, atol=5e-6)

        return

        # Test pdf and ppf ucensored
        for ucensor in [0.1, 0.5, 0.9]:
            pdfu = cop.pdf_ucensored(ucensor, uv[:, 1])

            expected = np.zeros_like(pdfu)
            pcensor = norm.ppf(ucensor)
            for i in range(nsamples):
                y = pq[i, 1]
                expected[i], err = quad(fun, -np.inf, pcensor, args=(y, ))


            spx = 0
            exp2 = 0
            for x in np.linspace(-5, pcensor, 100):
                px = norm.pdf(x)
                spx += px
                py = norm.pdf(pq[0, 1], loc=rho*x, scale=1-rho**2)
                exp2 += py*px
            exp2 /= spx
            import pdb; pdb.set_trace()


def test_gumbel(allclose):
    cop = copulas.GumbelCopula()



