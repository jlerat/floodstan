import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn, poisson, nbinom, bernoulli
from scipy.stats import multivariate_normal
from scipy.stats import ttest_1samp

import pytest
import warnings

from hydrodiy.plot import putils

import importlib
from tqdm import tqdm

from floodstan import marginals, sample, copulas
from floodstan import stan_test_marginal, stan_test_copula, \
                            stan_test_discrete

from test_sample_univariate import get_stationids, get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent


@pytest.mark.parametrize("marginal",
                         sample.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_marginals_vs_stan(marginal, stationid, allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    nboot = 100
    y = get_ams(stationid)
    N = len(y)
    dist = marginals.factory(marginal)

    for iboot in range(nboot):
        # Bootstrap fit
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, N)
        dist.params_guess(yboot)
        y0, y1 = dist.support

        sv = sample.StanSamplingVariable(yboot, marginal)
        sv.name = "y"
        stan_data = sv.to_dict()

        stan_data["ylocn"] = dist.locn
        stan_data["ylogscale"] = dist.logscale
        stan_data["yshape1"] = dist.shape1

        # Run stan
        smp = stan_test_marginal(data=stan_data)

        # Test
        luncens = smp.filter(regex="luncens").values
        expected = dist.logpdf(yboot)
        assert allclose(luncens, expected, atol=1e-5)

        cens = smp.filter(regex="^cens").values
        expected = dist.cdf(yboot)
        assert allclose(cens, expected, atol=1e-5)

        lcens = smp.filter(regex="^lcens").values
        expected = dist.logcdf(yboot)
        assert allclose(lcens, expected, atol=1e-5)


@pytest.mark.parametrize("copula",
                         sample.COPULA_NAMES_STAN)
def test_copulas_vs_stan(copula, allclose):
    N = 5000
    rng = np.random.default_rng(SEED)
    uv = rng.uniform(0, 1, size=(N, 2))
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    cop = copulas.factory(copula)

    rmin = cop.rho_min
    rmax = cop.rho_max
    nval = 20
    for rho in np.linspace(rmin, rmax, nval):
        cop.rho = rho

        stan_data = {
            "copula": sample.COPULA_NAMES[copula], \
            "N": N, \
            "uv": uv, \
            "rho": rho
        }

        # Run stan
        smp = stan_test_copula(data=stan_data)

        assert allclose(smp.rho_check, rho, atol=1e-6)

        # Test copula pdf
        lpdf = smp.filter(regex="luncens")
        expected = cop.logpdf(uv)
        assert allclose(lpdf, expected, atol=1e-7)

        # test copula cdf
        lcdf = smp.filter(regex="lcens")
        expected = cop.logcdf(uv)
        assert allclose(lcdf, expected, atol=1e-7)

        # Test copula conditional density
        lcond = smp.filter(regex="lcond")
        expected = np.log(cop.conditional_density(uv[:, 0], uv[:, 1]))
        assert allclose(lcond, expected, atol=1e-7)


@pytest.mark.parametrize("dname",
                         list(sample.DISCRETE_NAMES.keys()))
def test_discrete_vs_stan(dname, allclose):
    ntests = 10
    N = 100
    dcode = sample.DISCRETE_NAMES[dname]

    for i in range(ntests):
        locn_mu = 0.5 if dname == "Bernoulli" else 3
        locn_max = 1 if dname == "Bernoulli" else 20
        locn_sig = 0.1 if dname == "Bernoulli" else 1

        phi_mu, phi_sig = 1, 1

        p0, p1 = norm.cdf([0, locn_max], loc=locn_mu, scale=locn_sig)
        u = np.random.uniform()
        p = p0+(p1-p0)*u
        klocn = norm.ppf(p)*locn_sig+locn_mu

        p0, p1 = norm.cdf([1e-3, 3], loc=phi_mu, scale=phi_sig)
        u = np.random.uniform()
        p = p0+(p1-p0)*u
        kphi = norm.ppf(p, loc=phi_mu, scale=phi_sig)

        if dname == "Poisson":
            rcv = poisson(mu=klocn)
        elif dname == "Bernoulli":
            rcv = bernoulli(p=klocn)
        else:
            # reparameterize as per
            # https://mc-stan.org/docs/functions-reference/nbalt.html
            v = klocn+klocn**2/kphi
            n = klocn**2/(v-klocn)
            p = klocn/v
            rcv = nbinom(n=n, p=p)

        k = rcv.rvs(size=N).clip(0, sample.NEVENT_UPPER)
        kv = sample.StanDiscreteVariable(k, dname)
        stan_data = kv.to_dict()
        stan_data["klocn"] = klocn
        stan_data["kphi"] = kphi

        # Run stan
        smp = stan_test_discrete(data=stan_data)

        # Test
        lpmf = smp.filter(regex="lpmf").values
        expected = rcv.logpmf(k)
        assert allclose(lpmf, expected, atol=1e-5)


