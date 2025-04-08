#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-03-13 16:52:59.428042
## Comment : Explore continuing log pearson 3 pdf
##
## ------------------------------

import sys
import re
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils

from floodstan import marginals, sample
from floodstan import report, freqplots

from cmdstanpy import CmdStanModel

import importlib
importlib.reload(marginals)
importlib.reload(sample)

f = Path(sample.__file__).parent.parent.parent / "tests" /\
    "test_sample_univariate.py"
spec = importlib.util.spec_from_file_location("test_sample_univariate", f)
tsu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsu)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run logpearson 3 inference",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-c", "--censoring", help="Use censoring",
                    action="store_true", default=False)
args = parser.parse_args()
censoring = args.censoring

stationids = tsu.get_stationids()

nboot = 100

SEED = 5446

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "stan"
fout.mkdir(exist_ok=True, parents=True)

for f in fout.glob("*.*"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
LOGGER = sample.get_logger(stan_logger=False)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
marginal = marginals.factory("LogPearson3")

stan_file = froot / "scripts" / "logpearson3_check.stan"
LOGGER.info("Loading stan model")
model = CmdStanModel(stan_file=stan_file)
LOGGER.info(".. done")

for stationid in stationids:
    LOGGER.info(f"Station {stationid}")
    y = tsu.get_ams(stationid)
    N = len(y)
    ndone = 0

    for iboot in range(nboot):
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, N)

        censor = np.percentile(yboot, 20)
        dnocens = yboot[yboot >= censor]
        ncens = (yboot < censor).sum()

        # Run sampling variable with low number of
        # importance samples
        nimportance = 0
        sv = sample.StanSamplingVariable(marginal, yboot, censor,
                                 nimportance=nimportance,
                                 ninits=1)
        stan_data = sv.to_dict()
        marginal.params = sv.initial_parameters[0]

        # Test shape close to 0 for edge cases
        if iboot < nboot // 20:
            marginal.shape1 = 1e-20
        elif iboot >= nboot // 20 and iboot < 2 * nboot // 20:
            marginal.shape1 = 1e-3

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
        kwargs = dict()
        kwargs["data"] = stan_data
        kwargs["chains"] = 1
        kwargs["seed"] = SEED
        kwargs["iter_warmup"] = 1
        kwargs["iter_sampling"] = 1
        kwargs["fixed_param"] = True
        kwargs["show_progress"] = False
        fout_stan = fout / f"stan_{stationid}"
        for f in fout_stan.glob("*.*"):
            f.unlink()
        kwargs["output_dir"] = fout_stan
        out = model.sample(**kwargs)
        smp = out.draws_pd().squeeze()

        # Test params
        atol = 1e-5
        locn = smp.filter(regex="ylocn").values
        assert np.allclose(locn, marginal.locn, atol=atol)

        logscale = smp.filter(regex="ylogscale").values
        assert np.allclose(logscale, marginal.logscale, atol=atol)

        shape1 = smp.filter(regex="yshape1").values
        assert np.allclose(shape1, marginal.shape1, atol=atol)

        # Test data
        i11 = stan_data["i11"] - 1
        luncens = smp.filter(regex="luncens").values[i11]
        expected = marginal.logpdf(yboot[i11])
        assert np.allclose(luncens, expected, atol=atol)

        cens = smp.filter(regex="^cens").values[i11]
        expected = marginal.cdf(yboot[i11])
        assert np.allclose(cens, expected, atol=atol)

        atol = 5e-3
        lcens = smp.filter(regex="^lcens").values[i11]
        expected = marginal.logcdf(yboot[i11])
        assert np.allclose(lcens, expected, atol=atol)

        lpr = 0.
        atol = 1e-5
        for pn in marginals.PARAMETERS:
            lp = smp.filter(regex=f"logprior_{pn}")
            prior = getattr(marginal, f"{pn}_prior")
            expected = prior.logpdf(getattr(marginal, pn))
            assert np.allclose(lp, expected, atol=atol)
            lpr += lp.squeeze()

        ll = smp.filter(regex="loglikelihood").values[0]
        expected = marginal.logpdf(dnocens).sum()
        expected += ncens * marginal.logcdf(censor)
        assert np.allclose(ll, expected, atol=atol)

        lp = smp.filter(regex="logposterior").values[0]
        expected = -marginal.neglogpost(marginal.params, dnocens,
                                       censor, ncens)
        assert np.allclose(lp, expected, atol=atol)

    # Ensures at least 5 simulation beyond 0 shape trials
    assert ndone > nboot // 10 + 5

LOGGER.info("Process completed")
