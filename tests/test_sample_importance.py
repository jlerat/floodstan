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

from floodstan import marginals, sample
from floodstan import report

from test_sample_univariate import get_stationids, get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent

@pytest.mark.parametrize("stationid",
                         get_stationids() + ["hard"])
@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("censoring", [False, True])
def test_importance_sampling(stationid, distname, censoring):
    y = get_ams(stationid)
    censor = y.median() if censoring else -100.
    marginal = marginals.factory(distname)

    # First sample
    nsamples = 5000

    # .. use lh as prior
    params = sample.bootstrap(marginal, y, nboot=2000)
    # .. importance sampling
    smp, lps, neff = sample.importance_sampling(marginal, y, params,
                                           censor, nsamples)
    assert neff > 500

