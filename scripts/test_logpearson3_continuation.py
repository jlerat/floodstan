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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearson3
from scipy.special import gamma as gamma_fun

from hydrodiy.io import csv, iutils

from floodstan import marginals, sample
from floodstan import report

from cmdstanpy import CmdStanModel

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
distname = "LogPearson3"

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
LOGGER = sample.get_logger(stan_logger=True)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
# Set STAN
stan_nwarm = 10000
stan_nsamples = 5000
stan_nchains = 5

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

marginal = marginals.factory(distname)

locn = 40.
logscale = 0.5
scale = math.exp(logscale)
shape1 = 0.1

locn = math.log(10.) + 2 * scale / shape1
marginal.params = [locn, logscale, shape1]
LOGGER.info(str(marginal))

stan_data = {
    "ylocn": locn,
    "ylogscale": logscale,
    "yshape1": shape1
}

LOGGER.info("Run stan")
stan_file = froot / "scripts" / "logpearson3_test.stan"
model = CmdStanModel(stan_file=stan_file)

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
df = pd.DataFrame({re.sub('\^', '', pat): smp.filter(regex=pat).values
                   for pat in ["ydata", "lpdf", "lcdf"]})

LOGGER.info(f"smp alpha = {smp.alpha_copy} / {marginal.alpha}")
LOGGER.info(f"smp beta = {smp.beta_copy} / {marginal.beta}")
LOGGER.info(f"smp tau = {smp.tau_copy} / {marginal.tau}")

plt.close("all")
fig = plt.figure(figsize=(12, 6))
mosaic = [[cn for cn in df.columns if cn != "ydata"]]
axs = fig.subplot_mosaic(mosaic)
y = df.ydata
for varname, ax in axs.items():
    se = df.loc[:, varname]
    ax.plot(y, se, label=varname)
    tax = ax.twinx()
    tax.plot(y, se.diff(), color="k")
    ax.plot([], [], "k-", label=f"{varname} diff")
    ax.legend()
plt.show()

LOGGER.info("Process completed")
