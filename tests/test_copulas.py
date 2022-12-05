import json, re, math
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.stats import ks_2samp
from scipy.integrate import quad, nquad

from statsmodels.distributions.copula.api import GaussianCopula, \
                                            ClaytonCopula, \
                                            GumbelCopula, \
                                            FrankCopula
import pytest
import warnings

from nrivfloodfreqstan import copulas

#from tqdm import tqdm

FTESTS = Path(__file__).resolve().parent


# ---------------- UTITILITIES -------------------------------------------

# Dummy copula to test interface
class DummyCopula(copulas.Copula):
    def __init__(self):
        super(DummyCopula, self).__init__("Dummy")

    def ppf_conditional(self, ucond, q):
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
        assert rho>=0. and rho<=RHO_MAX, errmsg
        self._rho = rho

        if self.copula == "Gaussian":
            self._smcop = GaussianCopula(math.sin(math.pi*rho/2))
        elif self.copula == "Gumbel":
            self._smcop = GumbelCopula(1/(1-rho))
        elif self.copula == "Clayton":
            self._smcop = ClaytonCopula(2*rho/(1-rho))
        elif self.copula == "Frank":
            self._smcop = FrankCopula(2*rho/(1-rho))

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
    for copula in ["Gaussian", "Gumbel", "Clayton", "Frank"]:
        print(f"\nTesting {copula}")
        cop1 = SMCopula(copula)
        cop2 = copulas.factory(copula)
        #cop2.approx = False
        uv = np.random.uniform(0, 1, size=(ndata, 2))

        rhomax = 0.6 # theta<0.8 for Gaussian copula
        for rho in np.linspace(0.05, rhomax, 10):
            cop1.rho = rho
            cop2.rho = rho

            pdf1 = cop1.pdf(uv)
            pdf2 = cop2.pdf(uv)
            assert allclose(pdf1, pdf2)

            cdf1 = cop1.cdf(uv)
            cdf2 = cop2.cdf(uv)
            assert allclose(cdf1, cdf2, atol=1e-5)

            for itry in range(5):
                uv1 = cop1.sample(nsamples)
                uv2 = cop2.sample(nsamples)
                if copula == "Gaussian":
                    pq = norm.ppf(uv2)
                    corr = np.corrcoef(pq.T)[0, 1]
                    assert allclose(corr, cop2.theta, rtol=0, atol=1e-2)

                sa, pva = ks_2samp(uv1[:, 0], uv2[:, 0])
                sb, pvb = ks_2samp(uv1[:, 1], uv2[:, 1])
                if pva>0.2:
                    assert pvb>0.01

def test_plots():
    nsamples = 5000
    fimg = FTESTS / "images"
    fimg.mkdir(exist_ok=True)

    #for copula in ["Gaussian", "Gumbel", "Clayton", "Frank"]:
    for copula in ["Gumbel"]:
        cop = copulas.factory(copula)

        plt.close("all")
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), layout="tight")
        rhos = np.linspace(0.1, 0.6, 4)
        for iax, ax in enumerate(axs.flat):
            cop.rho = rhos[iax]
            samples = cop.sample(nsamples)
            samples[:, 0] = norm.ppf(samples[:, 0])
            samples[:, 1] = norm.ppf(samples[:, 1])
            ax.plot(samples[:, 0], samples[:, 1], ".", alpha=0.2)
            title = f"{copula} - rho={cop.rho:0.2f}"
            ax.set_title(title)

        fp = fimg / f"copula_sample_{copula}.png"
        fig.savefig(fp)



def test_gaussian(allclose):
    cop = copulas.GaussianCopula()

    ndata = 100
    uv = np.random.uniform(0, 1, size=(ndata, 2))
    pq = norm.ppf(uv)
    mu = np.zeros(2)

    for rho in np.linspace(0.1, 0.6, 6):
        cop.rho = rho
        theta = cop.theta

        # Test pdf
        pdf = cop.pdf(uv)
        Sigma = np.array([[1, theta], [theta, 1]])
        expected = multivariate_normal.pdf(pq, mean=mu, cov=Sigma)
        expected /= norm.pdf(pq[:,0])*norm.pdf(pq[:, 1])
        assert allclose(pdf, expected)

        s2 = np.sum(pq*pq, axis=1)
        z = s2-2*theta*np.prod(pq, axis=1)
        r2 = 1-theta*theta
        expected = np.exp(-z/r2/2+s2/2)/math.sqrt(r2)
        assert allclose(pdf, expected)

        # Test cdf
        cdf = cop.cdf(uv)

        def fun1(x, y):
            z = x*x-2*theta*x*y+y*y
            return math.exp(-z/r2/2)/2/math.pi/math.sqrt(r2)

        expected = np.zeros_like(cdf)
        for i in range(ndata):
            c1, c2 = pq[i]
            expected[i], err = nquad(fun1, [[-np.inf, c1], [-np.inf, c2]])

        assert allclose(cdf, expected, rtol=0, atol=5e-6)

        # Test conditional density
        def fun2(u, y):
            x = norm.ppf(u)
            z = x*x-2*theta*x*y+y*y
            return math.exp(-z/r2/2+(x*x+y*y)/2)/math.sqrt(r2)

        for ucond in np.linspace(0.1, 0.9, 5):
            pdfu = cop.conditional_density(ucond, uv[:, 1])

            expected = np.zeros_like(pdfu)
            for i in range(ndata):
                y = pq[i, 1]
                s, err = quad(fun2, 0, ucond, args=(y, ))
                expected[i] = s

            assert allclose(pdfu, expected, atol=1e-8)


def test_gumbel(allclose):
    cop = copulas.GumbelCopula()

    ndata = 100
    uv = np.random.uniform(0, 1, size=(ndata, 2))
    nsamples = 1000000

    for rho in np.linspace(0.1, 0.6, 6):
        cop.rho = rho
        theta = cop.theta
        samples = cop.sample(nsamples)

        x = -np.log(uv[:, 0])
        xth = np.power(x, theta)
        y = -np.log(uv[:, 1])
        yth = np.power(y, theta)

        # Test pdf
        pdf = cop.pdf(uv)
        expected = np.exp(-np.power(xth+yth, 1/theta))\
                    *(np.power(xth+yth, 1/theta)+theta-1)\
                    *np.power(xth+yth, 1/theta-2)\
                    *np.power(x*y, theta-1)\
                    *1/np.prod(uv, axis=1)
        assert allclose(pdf, expected)

        # Test cdf
        cdf = cop.cdf(uv)
        expected = np.exp(-np.power(xth+yth, 1/theta))
        assert allclose(cdf, expected, rtol=0, atol=5e-6)

        # Test conditional density
        def fun(u, v):
            return cop.pdf(np.array([u, v])[None, :])[0]

        for ucond in np.linspace(0.1, 0.9, 5):
            pdfu = cop.conditional_density(ucond, uv[:, 1])

            expected = np.zeros_like(pdfu)
            for i in range(ndata):
                s, err = quad(fun, 0, ucond, args=(uv[i, 1], ))
                expected[i] = s

            assert allclose(pdfu, expected, atol=1e-8)

        # Compare with reduced variables formula
        # Assumes x ~ GEV(kx, tx, ax) -> ux = F(x, kx, tx, ax)
        #                                   = exp{-[1-kx(x-tx)/ax]^(1/kx)}
        #         y ~ GEV(ky, ty, ay) -> uy = ...
        # Reduced variables are obtained as:
        #  wx = -1/kx log[1-kx(x-tx)/ax]
        #  wy = ...
        #
        # jy = -log(ay-ky*(y-ty))
        # a = exp(-theta*wx)+exp(-theta*wy)
        # logdensity_conditional = -a^(1/theta)+(1/theta-1)log(a)
        #                          - theta wy
        #                          + jy
        kx, tx, ax = -0.1, 100, 10
        ky, ty, ay = 0.1, 100, 20

        x = ax*(1-np.power(-np.log(uv[:, 0]), kx))/kx+tx
        y = ay*(1-np.power(-np.log(uv[:, 1]), ky))/ky+ty

        jy = -np.log(ay-ky*(y-ty))
        a = np.exp(-theta*wx)+np.exp(-theta*wy)
        import pdb; pdb.set_trace()

