import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import pytest
import warnings

from hydrodiy.plot import putils

import importlib
from tqdm import tqdm

from floodstan import marginals, sample, report
from floodstan import univariate_censored

from test_sample_univariate import get_stationids, get_ams, TQDM_DISABLE

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_report(allclose):
    stationid = "201001"
    marginal_name = "GEV"
    y = get_ams(stationid)
    N = len(y)

    marginal = marginals.factory(marginal_name)
    marginal.params_guess(y)

    # Prior marginalribution centered around marginal params
    ylocn_prior = [marginal.locn, abs(marginal.locn)*0.5]
    ylogscale_prior = [marginal.logscale, abs(marginal.logscale)*0.5]
    yshape1_prior = [max(0.1, marginal.shape1), 0.2]

    # Configure stan data and initialisation
    sv = sample.StanSamplingVariable(y, marginal_name)
    sv.name = "y"
    stan_data = sv.to_dict()
    stan_data["ylocn_prior"] = ylocn_prior
    stan_data["ylogscale_prior"] = ylogscale_prior
    stan_data["yshape1_prior"] = yshape1_prior

    # Clean output folder
    fout = FTESTS / "report" / stationid
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample
    smp = univariate_censored.sample(\
        data=stan_data, \
        chains=4, \
        seed=SEED, \
        iter_warmup=5000, \
        iter_sampling=500, \
        output_dir=fout, \
        inits=stan_data)

    # Get sample data
    params = smp.draws_pd()
    rep_df, rep_stat = report.ams_report(marginal, params)
    assert rep_stat.shape == (12, 8)

    obs = {year: y[year] for year in [1973, 2021]}
    rep_df, rep_stat = report.ams_report(marginal, params, obs)
    assert rep_stat.shape == (14, 8)

