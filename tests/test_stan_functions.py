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
from floodstan import stan_test_marginal, stan_test_copula

from test_sample_univariate import get_stationids, get_ams
from test_copulas import get_uv

SEED = 5446

FTESTS = Path(__file__).resolve().parent


@pytest.mark.parametrize("marginal",
                         sample.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_marginals_vs_stan(marginal, stationid, allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    nboot = 50
    y = get_ams(stationid)
    N = len(y)
    dist = marginals.factory(marginal)

    for iboot in range(nboot):
        # Bootstrap fit
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, N)
        dist.params_guess(yboot)

        # Test 0 shape for edge cases
        if marginal in ["GEV", "LogPearson3"]:
            if np.random.uniform(0, 1) < 0.1:
                dist.shape1 = 1e-20

        y0, y1 = dist.support

        sv = sample.StanSamplingVariable(yboot, marginal)
        stan_data = sv.to_dict()
        stan_data["ylocn"] = dist.locn
        stan_data["ylogscale"] = dist.logscale
        stan_data["yshape1"] = dist.shape1

        # Run stan
        smp = stan_test_marginal(data=stan_data)

        # Test
        locn = smp.filter(regex="ylocn").values
        assert allclose(locn, dist.locn, atol=1e-5)

        logscale = smp.filter(regex="ylogscale").values
        assert allclose(logscale, dist.logscale, atol=1e-5)

        shape1 = smp.filter(regex="yshape1").values
        assert allclose(shape1, dist.shape1, atol=1e-5)

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
    rng = np.random.default_rng(SEED)
    uv, N = get_uv()
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
        iok = np.isfinite(expected)
        assert allclose(lpdf[iok], expected[iok], atol=1e-7)

        # test copula cdf
        lcdf = smp.filter(regex="lcens")
        expected = cop.logcdf(uv)
        assert allclose(lcdf, expected, atol=1e-7)

        # Test copula conditional density
        lcond = smp.filter(regex="lcond")
        expected = np.log(cop.conditional_density(uv[:, 0], uv[:, 1]))
        assert allclose(lcond, expected, atol=1e-7)
