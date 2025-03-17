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

from scipy.stats import pearson3, gamma
from scipy.special import gamma as gamma_fun
from scipy.special import gammainc
from scipy.optimize import minimize_scalar, minimize

from hydrodiy.io import csv, iutils

from floodstan import marginals, sample
from floodstan import report

from cmdstanpy import CmdStanModel

import importlib
importlib.reload(marginals)

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

logscale = 0.5
scale = math.exp(logscale)
shape1 = 0.2
locn = math.log(10.) + 2 * scale / shape1
marginal.params = [locn, logscale, shape1]

LOGGER.info(str(marginal))

stan_data = {
    "N": 500,
    "ylower": 9.95,
    "yupper": 60000.,
    "ylocn": locn,
    "ylogscale": logscale,
    "yshape1": shape1
}

LOGGER.info("Run stan")
stan_file = froot / "scripts" / "logpearson3_test.stan"
fgamma_q_inv = stan_file.parent / "gamma_q_inv.hpp"
model = CmdStanModel(stan_file=stan_file,
                     user_header=fgamma_q_inv)

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
df = pd.DataFrame({re.sub("\^|.\\[", "", pat): smp.filter(regex=pat).values
                   for pat in ["ydata", "^lpdf", "^lcdf", "trans_pdf",
                               "trans_cdf", "^u\\["]})

LOGGER.info(f"smp alpha = {smp.alpha_copy} / {marginal.alpha:0.2f}")
LOGGER.info(f"smp beta = {smp.beta_copy} / {marginal.beta:0.2f}")
LOGGER.info(f"smp tau = {smp.tau_copy} / {marginal.tau:0.2f}")
abs_beta = abs(marginal.beta)
xth = math.exp(smp.lin_lpdf_trans / abs_beta + marginal.tau)
LOGGER.info(f"smp lin_lpdf_trans = {smp.lin_lpdf_trans} -> x = {xth:0.2f}")
xth = math.exp(smp.lin_lcdf_trans / abs_beta + marginal.tau)
LOGGER.info(f"smp lin_lcdf_trans = {smp.lin_lcdf_trans} -> x = {xth:0.2f}")
LOGGER.info(f"smp f0 = {smp.f0}")
LOGGER.info(f"smp ldf0 = {smp.ldf0}")

plt.close("all")
mosaic = [[cn for cn in df.columns if not re.search("ydata|^u|^trans", cn)]]
fig = plt.figure(figsize=(8 * len(mosaic[0]), 6),
                 layout="constrained")
axs = fig.subplot_mosaic(mosaic, sharex=True)
y = df.ydata
for varname, ax in axs.items():
    se = df.loc[:, varname]
    ax.plot(y, se, label=f"{varname} stan", lw=2)
    itrans = df.loc[:, f"trans_{varname[-3:]}"] == 1
    ax.plot(y[itrans], se[itrans], "+", color="0.3",
            alpha=0.5, zorder=0, label="Above tresh")

    fun = getattr(marginal, re.sub("l", "log", varname))
    se2 = fun(y)
    ax.plot(y, se2, label=f"{varname} scipy")

    tau = marginal.tau
    alpha = marginal.alpha
    abs_beta = abs(marginal.beta)
    sign_g = 1 if marginal.shape1 > 0 else -1
    ly = np.log(y)
    u = sign_g * (ly - tau) * abs_beta
    if varname == "lpdf":
        se3 = gamma.logpdf(u, a=alpha) + math.log(abs_beta) - ly
    elif varname == "lcdf":
        se3 = np.log(gammainc(alpha, u))
        LOGGER.info(f"min x valid = {y[np.isfinite(se3)].min():0.2f}")

    ax.plot(y, se3, label=f"{varname} scipy check")

    tax = ax.twinx()
    kw = dict(color="0.6", lw=0.8)
    #tax.plot(y, se.diff(), **kw)

    ax.set(ylabel=varname)
    tax.set(ylabel=f"Diff({varname})")
    ax.plot([], [], **kw, label=f"{varname} diff")
    ax.legend(loc=4)

plt.show()

LOGGER.info("Process completed")
