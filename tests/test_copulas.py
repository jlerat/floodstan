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

from tqdm import tqdm

FTESTS = Path(__file__).resolve().parent


# ---------------- UTITILITIES -------------------------------------------

# Dummy copula to test interface
class DummyCopula(copulas.Copula):
    def __init__(self):
        super(DummyCopula, self).__init__("Dummy")

    def ppf_conditional(self, ucond, q):
        return q


class SMCopula(copulas.Copula):
    def __init__(self, copula):
        super(SMCopula, self).__init__(f"SM-{copula}")
        self.copula = copula
        self.rho_min = 0.01
        self.rho_max = 0.95

    @copulas.Copula.rho.setter
    def rho(self, val):
        """ Set correlation parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{self.rho_min}, {self.rho_max}], got {rho}."
        assert rho>=self.rho_min and rho<=self.rho_max, errmsg
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
        cop1 = SMCopula(copula)
        cop2 = copulas.factory(copula)

        uv = np.random.uniform(0, 1, size=(ndata, 2))

        rmin = cop2.rho_min
        rmax = cop2.rho_max
        nval = 20
        tbar = tqdm(np.linspace(rmin, rmax, nval), \
                        desc=f"Testing "+copula+" vs statsmodels", total=nval)
        for rho in tbar:
            cop1.rho = rho
            cop2.rho = rho

            pdf1 = cop1.pdf(uv)
            pdf2 = cop2.pdf(uv)
            assert allclose(pdf1, pdf2, equal_nan=True, atol=1e8)

            cdf1 = cop1.cdf(uv)
            cdf2 = cop2.cdf(uv)
            assert allclose(cdf1, cdf2, atol=1e-5, equal_nan=True)

            # Statsmodels uses a random sample generator
            # that is not based in inverse CDF for Clayton
            if copula=="Clayton":
                continue

            nok = 0
            for itry in range(5):
                uv1 = cop1.sample(nsamples)
                uv2 = cop2.sample(nsamples)
                if copula == "Gaussian":
                    pq = norm.ppf(uv2)
                    corr = np.corrcoef(pq.T)[0, 1]
                    assert allclose(corr, cop2.theta, rtol=1e-2, atol=1e-2)

                sa, pva = ks_2samp(uv1[:, 0], uv2[:, 0])
                sb, pvb = ks_2samp(uv1[:, 1], uv2[:, 1])
                if pva>0.3:
                    nok += 1
                    assert pvb>0.01

            assert nok>0


def test_plots():
    nsamples = 5000
    fimg = FTESTS / "images"
    fimg.mkdir(exist_ok=True)

    for copula in ["Gaussian", "Gumbel", "Clayton", "Frank"]:
        cop = copulas.factory(copula)

        plt.close("all")
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), layout="tight")
        rhos = np.linspace(cop.rho_min, cop.rho_max, 4)
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
    for rho in np.linspace(cop.rho_min, cop.rho_max, 10):
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


def test_gumbel(allclose):
    cop = copulas.GumbelCopula()

    ndata = 100
    uv = np.random.uniform(0, 1, size=(ndata, 2))

    for rho in np.linspace(cop.rho_min, cop.rho_max, 10):
        cop.rho = rho
        theta = cop.theta

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
        assert allclose(pdf, expected, equal_nan=True)

        # Test cdf
        cdf = cop.cdf(uv)
        expected = np.exp(-np.power(xth+yth, 1/theta))
        assert allclose(cdf, expected, rtol=0, atol=5e-6)

        # Test conditional density
        def fun(u, v):
            return cop.pdf(np.array([u, v])[None, :])[0]

        tbar = tqdm(np.linspace(0.1, 0.9, 5), \
                desc=f"Testing gumbel rho={rho:0.3f}", total=5)
        for ucond in tbar:
            pdfu = cop.conditional_density(uv[:, 1], ucond)

            # Compare with reduced variables formula
            # Assumes x ~ GEV(kx, tx, ax) -> ux = F(x, kx, tx, ax)
            #                                   = exp{-[1-kx(x-tx)/ax]^(1/kx)}
            #         y ~ GEV(ky, ty, ay) -> uy = ...
            # Reduced variables are obtained as:
            #  wx = -1/kx log[1-kx(x-tx)/ax]
            #  wy = ...
            #
            # jacy = -log(ay-ky*(y-ty))
            # a = exp(-theta*wx)+exp(-theta*wy)
            # logdensity_conditional = -a^(1/theta)+(1/theta-1)log(a)
            #                          - theta wy
            #                          + jacy
            kx, tx, ax = -0.1, 100, 90
            ky, ty, ay = 0.1, 100, 50

            xcens = ax*(1-np.power(-np.log(ucond), kx))/kx+tx
            y = ay*(1-np.power(-np.log(uv[:, 1]), ky))/ky+ty

            wxcens = -1/kx*np.log(1-kx*(xcens-tx)/ax)
            wy = -1/ky*np.log(1-ky*(y-ty)/ay)

            logjacy = -np.log(ay-ky*(y-ty))
            a = np.exp(-theta*wxcens)+np.exp(-theta*wy)
            expected = -np.power(a, 1/theta)+(1/theta-1)*np.log(a)-theta*wy+logjacy

            # Direct expression from copula
            eta = 1-ky*(y-ty)/ay
            fy = 1/ay*np.exp(-np.power(eta, 1/ky))*np.power(eta, 1/ky-1)
            lpdf = np.log(fy)+np.log(pdfu)
            iok = np.isfinite(lpdf)
            assert allclose(lpdf[iok], expected[iok])



def test_conditional_density(allclose):
    ndata = 100
    for copula in ["Gaussian", "Gumbel", "Clayton", "Frank"]:
        cop = copulas.factory(copula)
        uv = np.random.uniform(0, 1, size=(ndata, 2))

        for rho in np.linspace(cop.rho_min, cop.rho_max, 10):
            cop.rho = rho

            nval = 10
            tbar = tqdm(np.linspace(0.1, 0.9, nval), \
                    desc=f"Testing {copula} conditional density: rho={rho:0.3f}", \
                    total=nval)
            for ucond in tbar:
                pdfu = cop.conditional_density(uv[:, 1], ucond)
                assert (pdfu<1).sum()>0

                # test pdfu is reverse of ppf conditional
                # except for Gumbel which does not use
                # ppf_conditional for the sample function
                if copula != "Gumbel":
                    iok = pdfu<0.99
                    qu = cop.ppf_conditional(uv[iok, 1], pdfu[iok])
                    assert allclose(qu, ucond, atol=1e-3)

                # Test conditional density vs integrating pdf
                expected1 = np.zeros_like(pdfu)
                def fun(u, v):
                    return cop.pdf(np.array([u, v])[None, :])[0]

                for i in range(ndata):
                    s, err = quad(fun, 0, ucond, args=(uv[i, 1], ))
                    expected1[i] = s

                assert allclose(pdfu, expected1, atol=1e-4)

                # Test conditional density vs derivating cdf
                uvc = uv.copy()
                uvc[:, 0] = ucond
                c0 = cop.cdf(uvc)
                eps = 1e-6
                uvc[:, 1]+=eps
                c1 = cop.cdf(uvc)
                expected2 = (c1-c0)/eps
                assert allclose(pdfu, expected2, atol=5e-4, rtol=5e-4)


