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
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils

from floodstan import marginals, sample
from floodstan import report, freqplots

from cmdstanpy import CmdStanModel

import importlib
importlib.reload(marginals)

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
dist = marginals.factory("LogPearson3")

stan_file = froot / "scripts" / "logpearson3_check.stan"
model = CmdStanModel(stan_file=stan_file)
print(model)

for stationid in stationids:
    LOGGER.info(f"Station {stationid}")
    y = tsu.get_ams(stationid)

    for iboot in range(nboot):
        # Bootstrap fit
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, len(y))
        dist.params_guess(yboot)

        # Test 0 shape for edge cases
        if np.random.uniform(0, 1) < 0.1:
            dist.shape1 = 1e-20

        sv = sample.StanSamplingVariable(dist, yboot)
        stan_data = sv.to_dict()
        stan_data["ylocn"] = dist.locn
        stan_data["ylogscale"] = dist.logscale
        stan_data["yshape1"] = dist.shape1

        kwargs = {}
        kwargs["chains"] = 1
        kwargs["iter_warmup"] = 1
        kwargs["iter_sampling"] = 1
        kwargs["fixed_param"] = True
        kwargs["show_progress"] = False
        kwargs["show_progress"] = False
        kwargs["output_dir"] = fout
        fit = model.sample(data=stan_data, **kwargs)
        smp = fit.draws_pd().squeeze()

        # Test
        errmess = str(dist)
        tau = smp.filter(regex="tau_copy").values
        assert np.allclose(tau, dist.tau, atol=1e-5), errmess

        beta = smp.filter(regex="beta_copy").values
        assert np.allclose(beta, dist.beta, atol=1e-5), errmess

        alpha = smp.filter(regex="alpha_copy").values
        assert np.allclose(alpha, dist.alpha, atol=1e-5), errmess

        luncens = smp.filter(regex="luncens").values
        expected = dist.logpdf(yboot)
        assert np.allclose(luncens, expected, atol=1e-5), errmess

        cens = smp.filter(regex="^cens").values
        expected = dist.cdf(yboot)
        assert np.allclose(cens, expected, atol=1e-5), errmess

        lcens = smp.filter(regex="^lcens").values
        expected = dist.logcdf(yboot)
        assert np.allclose(lcens, expected, atol=1e-5), errmess


LOGGER.info("Process completed")
