#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-03-31 12:15:21.000583
## Comment : Check computation of incomplete gamma
##
## ------------------------------

import sys
import re
import math
from itertools import product as prod
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
from scipy.special import erfc

import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel

from hydrodiy.io import iutils
from hydrodiy.plot import putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "check_incomplete_gamma"
fout.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

stan_file = froot / "scripts" / "gammacdf_check.stan"
model = CmdStanModel(stan_file=stan_file)

n = 10
ng, nsigs = n, n
gs = np.logspace(-4, -2, ng)
sigs = np.logspace(-1, 1, nsigs)

G, S = np.meshgrid(gs, sigs)
E2 = np.zeros_like(G)
E3 = np.zeros_like(G)

for idxg, idxs in prod(range(n), range(n)):
    g = G[idxg, idxs]
    s = S[idxg, idxs]

    alpha = 4. / g / g
    beta = 2. / g / s

    y = np.logspace(0, 3, 500)
    ly = np.log(y)
    m = ly.mean()
    tau = m - alpha / beta
    ly = ly[ly > tau]
    x = ly - tau
    stan_data = {
        "x": x,
        "N": len(x),
        "alpha": alpha,
        "beta": beta
    }
    kwargs = {}
    kwargs["chains"] = 1
    kwargs["iter_warmup"] = 1
    kwargs["iter_sampling"] = 1
    kwargs["fixed_param"] = True
    kwargs["show_progress"] = False
    kwargs["output_dir"] = fout
    fit = model.sample(data=stan_data, **kwargs)
    smp = fit.draws_pd().squeeze()
    y1 = smp.filter(regex="cdf").values

    # Scipy computations
    y2 = gamma.cdf(x, a=alpha, scale=1./beta)
    y3 = norm.cdf(ly, loc=m, scale=s)

    E2[idxg, idxs] = np.abs(y2 - y1).max()
    E3[idxg, idxs] = np.abs(y2 - y3).max()

plt.close("all")
fig, axs = plt.subplots(ncols=2,
                        figsize=(18, 6),
                        layout="constrained")

for iax, (ax, E) in enumerate(zip(axs, [E2, E3])):
    cnf = ax.contourf(G, S, E)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.colorbar(cnf)

    if iax == 0:
        title = "Diff stan gamma - scipy gamma"
    else:
        title = "Diff scipy norm - scipy gamma"
    ax.set(title=title, xlabel="g", ylabel="s")

plt.show()
LOGGER.completed()

