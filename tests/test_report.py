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
from floodstan import univariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams, TQDM_DISABLE

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent

def test_process_stan_diagnostic():
    fd = FTESTS / "stan_diag.txt"
    with fd.open("r") as fo:
        diag = fo.read()

    dd = report.process_stan_diagnostic(diag)
    kk = ["message", "treedepth", "divergence", "ebfmi", "effsamplesz", "rhat"]
    for k in kk:
        assert k in dd


def test_report(allclose):
    stationid = "201001"
    marginal_name = "GEV"
    y = get_ams(stationid)
    N = len(y)

    # Configure stan data and initialisation
    sv = sample.StanSamplingVariable(y, marginal_name)
    stan_data = sv.to_dict()

    # Clean output folder
    fout = FTESTS / "report" / stationid
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample
    smp = univariate_censored_sampling(\
        data=stan_data, \
        chains=4, \
        seed=SEED, \
        iter_warmup=5000, \
        iter_sampling=500, \
        output_dir=fout, \
        inits=stan_data)

    # Get sample data
    params = smp.draws_pd()

    rep, _ = report.ams_report(sv.marginal, params)
    assert rep.shape == (12, 11)

    obs = {year: y[year] for year in [1973, 2021]}
    rep, _ = report.ams_report(sv.marginal, params, obs)
    assert rep.shape == (16, 11)

