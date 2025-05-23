import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod
import warnings

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


@pytest.mark.parametrize("marginal_name",
                         sample.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_marginals_vs_stan(marginal_name, stationid, allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    nboot = 100
    y = get_ams(stationid)
    N = len(y)
    marginal = marginals.factory(marginal_name)
    ndone = 0
    for iboot in range(nboot):
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, N)

        censor = np.percentile(yboot, 20)
        dnocens = yboot[yboot >= censor]
        ncens = (yboot < censor).sum()

        sv = sample.StanSamplingVariable(marginal, yboot, censor,
                                         ninits=1)
        stan_data = sv.to_dict()
        marginal.params = {k[1:]: v for k, v in
                           sv.initial_parameters[0].items()}
        # Test shape close to 0 for edge cases
        if marginal.has_shape:
            if iboot < nboot // 20:
                marginal.shape1 = 1e-20
                # .. fix prior if needed
                l = stan_data["shape1_lower"]
                stan_data["shape1_lower"] = min(l, 1e-20)
                u = stan_data["shape1_upper"]
                stan_data["shape1_upper"] = max(l, 1e-20)

            elif iboot >= nboot // 20 and iboot < 2 * nboot // 20:
                marginal.shape1 = 1e-3
                # .. fix prior if needed
                l = stan_data["shape1_lower"]
                stan_data["shape1_lower"] = min(l, 1e-3)
                u = stan_data["shape1_upper"]
                stan_data["shape1_upper"] = max(l, 1e-3)

        y0, y1 = marginal.support
        ynocens = yboot[yboot > censor]
        if y0 > ynocens.min() or y1 < ynocens.max():
            wmess = "Skipping because data is outside of support"
            warnings.warn(wmess)
            continue

        ndone += 1
        stan_data["ylocn"] = marginal.locn
        stan_data["ylogscale"] = marginal.logscale
        stan_data["yshape1"] = marginal.shape1

        # Run stan
        smp = stan_test_marginal(data=stan_data)

        # Test params
        atol = 5e-4
        locn = smp.filter(regex="ylocn").values
        assert allclose(locn, marginal.locn, atol=atol)

        logscale = smp.filter(regex="ylogscale").values
        assert allclose(logscale, marginal.logscale, atol=atol)

        shape1 = smp.filter(regex="yshape1").values
        assert allclose(shape1, marginal.shape1, atol=atol)

        # Test data
        i11 = stan_data["i11"] - 1
        atol = 5e-3
        luncens = smp.filter(regex="luncens").values[i11]
        expected = marginal.logpdf(yboot[i11])
        assert allclose(luncens, expected, atol=atol)

        cens = smp.filter(regex="^cens").values[i11]
        expected = marginal.cdf(yboot[i11])
        assert allclose(cens, expected, atol=atol)

        lcens = smp.filter(regex="^lcens").values[i11]
        expected = marginal.logcdf(yboot[i11])
        assert allclose(lcens, expected, atol=atol)

        lpr = 0.
        atol = 1e-5
        for pn in marginals.PARAMETERS:
            lp = smp.filter(regex=f"logprior_{pn}")
            prior = getattr(marginal, f"{pn}_prior")

            # .. need to adjust prior
            prior.lower = stan_data[f"{pn}_lower"]
            prior.upper = stan_data[f"{pn}_upper"]
            prior.loc = stan_data[f"y{pn}_prior"][0]
            prior.scale = stan_data[f"y{pn}_prior"][1]

            expected = prior.logpdf(getattr(marginal, pn))
            assert allclose(lp, expected, atol=atol)
            lpr += lp.squeeze()

        ll = smp.filter(regex="loglikelihood").values[0]
        expected = marginal.logpdf(dnocens).sum()
        expected += ncens * marginal.logcdf(censor)
        err = abs(math.asinh(ll) - math.asinh(expected))
        assert err < 1e-3

        lp = smp.filter(regex="logposterior").values[0]
        expected = -marginal.neglogpost(marginal.params, dnocens,
                                       censor, ncens)
        err = abs(math.asinh(lp) - math.asinh(expected))
        assert err < 1e-3

    # Ensures at least 5 simulation beyond 0 shape trials
    assert ndone > nboot // 10 + 5


@pytest.mark.parametrize("copula",
                         sample.COPULA_NAMES_STAN)
def test_copulas_vs_stan(copula, allclose):
    rng = np.random.default_rng(SEED)
    uv, N = get_uv()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    cop = copulas.factory(copula)

    rmin = cop.rho_lower
    rmax = cop.rho_upper
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
