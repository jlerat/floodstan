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
from tqdm import tqdm

from floodstan import marginals
from floodstan import stan_test_marginal, stan_test_copula

from floodstan import report, sample
from floodstan import bivariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent

STATIONIDS = get_stationids()

@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid", STATIONIDS)
@pytest.mark.parametrize("fit_method", ["params_guess", "fit_lh_moments"])
def test_bootstrap(distname, stationid, fit_method):
    y = get_ams(stationid)
    marginal = marginals.factory(distname)
    if distname == "Gamma" and fit_method == "fit_lh_moments":
        pytest.skip("Lh moments not available for Gamma.")

    boot = sample.univariate_bootstrap(marginal, y, fit_method=fit_method,
                            nboot=1000)

    nok = boot.notnull().all(axis=1).sum()
    assert nok > len(boot) * 0.9

    std = boot.std()
    assert std.locn > 0
    assert std.logscale > 0
    if marginal.has_shape:
        assert std.shape1 > 0

