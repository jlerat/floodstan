import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import pytest
import warnings

import importlib

from floodstan import marginals, sample, report
from floodstan import univariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent

def test_process_stan_diagnostic():
    fd = FTESTS / "stan_diag.txt"
    with fd.open("r") as fo:
        diag = fo.read()

    dd = report.process_stan_diagnostic(diag)
    kk = ["message", "treedepth", "divergence", "ebfmi", "effsamplesz", "rhat",
          "divergence_proportion"]
    for k in kk:
        assert k in dd

    assert dd["divergence_proportion"] == 0


def test_report(allclose):
    stationid = "201001"
    marginal_name = "GEV"
    y = get_ams(stationid)
    N = len(y)
    stan_nchains = 5

    # Configure stan data and initialisation
    sv = sample.StanSamplingVariable(y, marginal_name,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # Clean output folder
    LOGGER = sample.get_logger(stan_logger=False)
    fout = FTESTS / "report" / stationid
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample
    smp = univariate_censored_sampling(data=stan_data,
                                       chains=stan_nchains,
                                       seed=SEED,
                                       iter_warmup=5000,
                                       iter_sampling=500,
                                       output_dir=fout,
                                       inits=stan_inits)
    # Get sample data
    params = smp.draws_pd()

    # Run report without obs
    rep, _ = report.ams_report(sv.marginal, params) #, design_aris=[100])
    assert rep.shape == (12, 12)

    # Run report with obs
    years = np.arange(1973, 2022)
    obs = {year: y[year] for year in years}
    rep, _ = report.ams_report(sv.marginal, params, obs)
    assert rep.shape == (12 + 2 * len(obs), 12)

