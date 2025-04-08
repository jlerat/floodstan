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

    for iboot in range(nboot):
        # Bootstrap fit
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, len(y))
        marginal.params_guess(yboot)

        # exclude if support to too narrow
        y0, y1 = marginal.support
        if y0 > yboot.min() or y1 < yboot.max():
            continue

        # Test 0 shape for edge cases
        if iboot < nboot // 20:
            marginal.shape1 = 1e-20

        if iboot > nboot // 20 & iboot < nboot // 10:
            marginal.shape1 = 1e-3

        sv = sample.StanSamplingVariable(marginal, yboot,
                                         ninits=1)
        stan_data = sv.to_dict()
        stan_data["ylocn"] = marginal.locn
        stan_data["ylogscale"] = marginal.logscale
        stan_data["yshape1"] = marginal.shape1

        for f in fout.glob("*.*"):
            f.unlink()

        kwargs = {}
        kwargs["chains"] = 1
        kwargs["iter_warmup"] = 1
        kwargs["iter_sampling"] = 1
        kwargs["fixed_param"] = True
        kwargs["show_progress"] = False
        kwargs["show_progress"] = False
        kwargs["output_dir"] = fout
        kwargs["seed"] = SEED
        fit = model.sample(data=stan_data, **kwargs)
        smp = fit.draws_pd().squeeze()

        # Test
        errmess = f"Error using {marginal}."

        tau = smp.filter(regex="tau").values
        assert np.allclose(tau, marginal.tau, atol=1e-5), errmess

        beta = smp.filter(regex="beta").values
        assert np.allclose(beta, marginal.beta, atol=1e-5), errmess

        alpha = smp.filter(regex="alpha").values
        assert np.allclose(alpha, marginal.alpha, atol=1e-5), errmess

        luncens = smp.filter(regex="luncens").values
        expected = marginal.logpdf(yboot)
        assert np.allclose(luncens, expected, atol=1e-5), errmess

        cens = smp.filter(regex="^cens").values
        expected = marginal.cdf(yboot)
        assert np.allclose(cens, expected, atol=1e-5), errmess

        lcens = smp.filter(regex="^lcens").values
        expected = marginal.logcdf(yboot)
        assert np.allclose(lcens, expected, atol=1e-5), errmess


LOGGER.info("Process completed")
