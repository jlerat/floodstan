#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-03-13 16:52:59.428042
## Comment : Explore genlogistic
##
## ------------------------------


import sys
import re
import json
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

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
distname = "GeneralizedLogistic"
marginal = marginals.factory(distname)

aris = np.array([2, 5, 10, 20, 50, 100, 500, 1000])

stan_nchains = 3
stan_nwarm = 100
nsamples = 100

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata = froot / "tests" / "data" / "bivariate_GeneralizedLogistic"

fout = froot / "outputs" / "genlogistic_fix"
fout.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
LOGGER = sample.get_logger(stan_logger=True)
#_ = sample.get_logger(stan_logger=False)
#basename = source_file.stem
#LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
#fargs = fdata / "stan_args_ok.json"
fargs = fdata / "stan_args.json"
with fargs.open("r") as fo:
    stan_args = json.load(fo)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
LOGGER.info("Stan compilation")
stan_file = source_file.parent / "genlogistic_run.stan"
model = CmdStanModel(stan_file=stan_file)
print(model)

LOGGER.info("Stan data build")
fout_stan = fout / "stan"
fout_stan.mkdir(exist_ok=True)
for f in fout_stan.glob("*.*"):
    f.unlink()
stan_args["output_dir"] = fout_stan

stan_args["chains"] = stan_nchains
stan_args["iter_warmup"] = stan_nwarm
stan_args["iter_warmup"] = nsamples // stan_nchains
stan_args["show_progress"] = True

stan_args.pop("inits")

LOGGER.info("Stan sampling")
fit = model.sample(**stan_args)

LOGGER.info("Stan postprocessing")
smp = fit.draws_pd().squeeze()
diag = report.process_stan_diagnostic(fit.diagnose())
for n in ["divergence", "ebfmi", "effsamplesz", "rhat"]:
    LOGGER.info(f"Diag {n} : {diag[n]}", ntab=1)

rep, _ = report.ams_report(sv.marginal, smp, design_aris=aris)

#for f in fout_stan.glob("*.*"):
#    f.unlink()
#fout_stan.rmdir()

LOGGER.info("Plotting", ntab=1)
plt.close("all")
fig, axs = plt.subplots(ncols=1,
                        figsize=(16, 8),
                        layout="constrained")
ptype = "gumbel"
quantiles = rep.filter(regex="DESIGN", axis=0)
ax = axs[0]
freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                  center_column="MEAN",
                                  q0_column="5%",
                                  q1_column="95%",
                                  label="GeneralizedLogistic - stan",
                                  alpha=0.3,
                                  facecolor="tab:blue",
                                  edgecolor="k")
freqplots.plot_data(ax, y, ptype)
retp = [5, 10, 100, 500]
aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

fig.suptitle(f"Station {stationid}")
fig.savefig(fout / f"{stationid}_ffa.png")

LOGGER.info("Process completed")
