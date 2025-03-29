#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-03-28 16:27:15.565227
## Comment : Compare different quantile estimators
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

from floodstan import marginals
from floodstan import sample
from floodstan import report
from floodstan import univariate_censored_sampling
from floodstan import quadapprox

source_file = Path(__file__)
froot = source_file.parent.parent
import importlib
spec = importlib.util.spec_from_file_location("test_sample_univariate",
                                              froot / "tests" / "test_sample_univariate.py")
test_sample_univariate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_sample_univariate)

importlib.reload(report)
importlib.reload(quadapprox)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
stationids = test_sample_univariate.get_stationids()

marginal = marginals.factory("GEV")

stan_nwarm = 10000
stan_nsamples = 10000
stan_nchains = 10

seed = 5446

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
fout = froot / "outputs" / "quantile_estimators"
fout.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

for stationid in stationids:
    y = test_sample_univariate.get_ams(stationid)
    censor = y.quantile(0.3)

    sv = sample.StanSamplingVariable(marginal, y, censor,
                                     ninits=stan_nchains)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    # Clean output folder
    fout_stan = fout / f"stan_{stationid}"
    fout_stan.mkdir(exist_ok=True)
    for f in fout_stan.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=seed,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout_stan,
              inits=stan_inits,
              chains=stan_nchains,
              iter_warmup=stan_nwarm)

    # Sample
    smp = univariate_censored_sampling(**kw)
    params = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())
    rep, _ = report.ams_report(marginal, params)

    # Clean
    for f in fout_stan.glob("*.*"):
        f.unlink()
    fout_stan.rmdir()

    # pred dist
    pred = report.predictive_distribution(marginal, params)

    # Mean params
    cc = [f"y{n}" for n in marginals.PARAMETERS]
    marginal.params = params.loc[:, cc].mean()
    aris = report.DESIGN_ARIS
    cdf = 1 - 1./np.array(aris)
    meanp = pd.DataFrame({"MP": marginal.ppf(cdf)},
                         index=pred.index)

    # plot
    plt.close("all")
    fig, ax = plt.subplots()
    ptype = "gumbel"
    freqplots.plot_data(ax, y, ptype)

    expected = rep.filter(regex="DESIGN", axis=0)
    freqplots.plot_marginal_quantiles(ax, aris, expected, ptype,
                                      center_column="MEAN",
                                      label="GEV - expected", color="red")

    freqplots.plot_marginal_quantiles(ax, aris, pred, ptype,
                                      center_column="PREDICTIVE",
                                      label="GEV - predictive", color="green")

    freqplots.plot_marginal_quantiles(ax, aris, meanp.loc[:, ["MP"]], ptype,
                                      center_column="MP",
                                      label="GEV - mean params", color="blue")
    retp = [5, 10, 100, 500]
    aeps, xpos = freqplots.add_aep_to_xaxis(ax, ptype, retp)

    ax.legend()
    ax.set(title = stationid)
    fp = fout / f"quantile_estimators_{stationid}.png"
    fig.savefig(fp)

LOGGER.completed()

