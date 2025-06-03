import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.stats import ttest_1samp

import pytest
import warnings

from hydrodiy.plot import putils

import importlib

from scipy.stats import multivariate_normal as mvt
from scipy.stats import cramervonmises

from floodstan import marginals, sample
from floodstan import copulas, report

from test_sample_univariate import get_stationids, get_ams
from test_sample_bivariate import add_gaussian_covariate

SEED = 5446

FTESTS = Path(__file__).resolve().parent

def test_generic_importance_sampling(allclose):
    nvars = 2
    nparams = nvars + (nvars + 1) * nvars // 2
    nrepeat = 50
    ndata = 100
    nsamples = 500

    mu = np.linspace(0, 2, nvars)
    m = np.random.uniform(-1, 1, size=(nvars, nvars))
    U, S, Vt = np.linalg.svd(m)
    cov = U@np.diag(np.linspace(1, 2, nvars))@U.T

    iu0 = np.triu_indices(nvars)
    iu1 = np.triu_indices(nvars, 1)
    il1 = np.tril_indices(nvars, -1)
    theta = np.concatenate([mu, cov[iu0].ravel()])

    cov_lp = np.zeros((nvars, nvars))
    pits = np.zeros((nrepeat, nparams))

    for i in range(nrepeat):
        x = np.random.multivariate_normal(mean=mu, cov=cov,
                                          size=ndata)
        def logpost(theta):
            mu = theta[:nvars]
            cov_lp[iu0] = theta[nvars:]
            cov_lp[il1] = cov_lp.T[il1]
            return mvt.logpdf(x, mean=mu, cov=cov_lp).sum()

        eps = np.random.uniform(-0.1, 0.1, size=(nsamples, nparams))
        params = theta[None, :] + eps

        params, logposts, neff, niter = \
            sample.generic_importance_sampling(params, logpost, nsamples)

        pits[i] = (params - theta >= 0).sum() / nsamples

    u = np.linspace(0, 1, nrepeat)
    dist = np.max(np.abs(np.sort(pits, axis=0) - u[:, None]), axis=0)
    assert dist.max() < 0.25


@pytest.mark.parametrize("stationid",
                         get_stationids()[:2] + ["hard"])
@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("censoring", [False, True])
def test_univariate_importance_sampling(stationid, distname, censoring):
    y = get_ams(stationid)
    censor = y.median() if censoring else -100.
    marginal = marginals.factory(distname)

    nsamples = 1000
    smp, lps, neff, niter = sample.univariate_importance_sampling(marginal, y,
                                                                  censor,
                                                                  nsamples)
    print(f"univariate - neff={neff:0.0f}/{nsamples} niter={niter}")
    assert neff > 200
    assert lps.std() > 0
    assert (smp.std().iloc[:2] > 0).all()


@pytest.mark.parametrize("stationid",
                         get_stationids()[:2] + ["hard"])
@pytest.mark.parametrize("copula", sample.COPULA_NAMES_STAN)
@pytest.mark.parametrize("censoring", [True])
def test_bivariate_importance_sampling(stationid, copula, censoring, allclose):
    LOGGER = sample.get_logger(stan_logger=True)

    marginaly = marginals.factory("GEV")
    marginalz = marginals.factory("GEV")
    cop = copulas.factory(copula)

    y = get_ams(stationid)
    y, z = add_gaussian_covariate(y)
    data = np.column_stack([y, z])

    ycensor = y.median() if censoring else np.nanmin(y) - 1.
    zcensor = z.median() if censoring else np.nanmin(z) - 1.

    nsamples = 1000
    smp, lps, neff, niter = sample.bivariate_importance_sampling(marginaly,
                                                                 marginalz,
                                                                 cop,
                                                                 data,
                                                                 ycensor,
                                                                 zcensor,
                                                                 nsamples)

    print(f"bivariate - neff={neff:0.0f}/{nsamples} niter={niter}")
    assert neff > 200
    assert lps.std() > 0
    assert (smp.std().iloc[:2] > 0).all()



