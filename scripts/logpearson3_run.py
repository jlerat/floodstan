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

distname = "LogPearson3"
marginal = marginals.factory(distname)

marginal.shape1_prior.lower = -1.9
marginal.shape1_prior.upper = 1.9

stan_nwarm = 5000
stan_nsamples = 5000
stan_nchains = 5

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "logpearson3_fix"
fout.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
_ = sample.get_logger(stan_logger=False)
LOGGER = iutils.get_logger(basename)
LOGGER.info(f"Censoring : {censoring}")

for stationid in stationids:
    LOGGER.info(f"Processing {stationid}")

    # ----------------------------------------------------------------------
    # @Get data
    # ----------------------------------------------------------------------
    y = tsu.get_ams(stationid)
    censor = y.median() if censoring else np.nanmin(y) - 1e-3
    sv = sample.StanSamplingVariable(marginal, y, censor)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # ----------------------------------------------------------------------
    # @Process
    # ----------------------------------------------------------------------

    stan_file = froot / "scripts" / "logpearson3_run.stan"
    model = CmdStanModel(stan_file=stan_file)
    print(model)

    # sample
    kwargs = {}
    kwargs["chains"] = stan_nchains
    kwargs["parallel_chains"] = stan_nchains
    kwargs["iter_warmup"] = stan_nwarm
    kwargs["iter_sampling"] = stan_nsamples // stan_nchains
    kwargs["show_progress"] = False

    fout_stan = fout / f"stan_{stationid}"
    for f in fout_stan.glob("*.*"):
        f.unlink()

    kwargs["output_dir"] = fout_stan

    LOGGER.info("Sampling", ntab=1)
    fit = model.sample(data=stan_data, **kwargs)

    LOGGER.info("Processing", ntab=1)
    smp = fit.draws_pd().squeeze()
    diag = report.process_stan_diagnostic(fit.diagnose())
    for n in ["divergence", "ebfmi", "effsamplesz", "rhat"]:
        LOGGER.info(f"Diag {n} : {diag[n]}", ntab=1)

    aris = np.array([2, 5, 10, 20, 50, 100, 500, 1000])
    rep, _ = report.ams_report(sv.marginal, smp, design_aris=aris)

    for f in fout_stan.glob("*.*"):
        f.unlink()
    fout_stan.rmdir()

    LOGGER.info("Plotting", ntab=1)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8),
                           layout="constrained")
    ptype = "gumbel"
    quantiles = rep.filter(regex="DESIGN", axis=0)
    freqplots.plot_marginal_quantiles(ax, aris, quantiles, ptype,
                                      center_column="MEAN",
                                      q0_column="5%",
                                      q1_column="95%",
                                      label="LogPearson3",
                                      alpha=0.3,
                                      facecolor="tab:blue",
                                      edgecolor="k")
    freqplots.plot_data(ax, y, ptype)

    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    fig.suptitle(f"Station {stationid}")
    fig.savefig(fout / f"{stationid}_ffa.png")

    sys.exit()

LOGGER.info("Process completed")
