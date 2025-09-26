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
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils

from floodstan import marginals, sample
from floodstan import report, freqplots

from cmdstanpy import CmdStanModel

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
nboot = 20
nboot_zero = 3

SEED = 5446

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fdata = froot / "tests" / "data"

fout = froot / "outputs" / "genlogistic_fix" / "checks"
fout.mkdir(exist_ok=True, parents=True)

fout_stan = fout / "stan"

for f in fout_stan.glob("*.*"):
    f.unlink()

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
LOGGER = sample.get_logger(stan_logger=False)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
ams = {}
for f in  fdata.glob("*.csv"):
    stationid = re.sub("_AMS.*", "", f.stem)
    if not re.search("\d{6}", stationid):
        continue
    df = pd.read_csv(f, skiprows=15)
    ams[stationid] = df.iloc[:, 1]

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
marginal = marginals.factory("GeneralizedLogistic")

LOGGER.info("Compiling stan model")
stan_file = froot / "scripts" / "generalizedLogistic" / "genlogistic_check.stan"
model = CmdStanModel(stan_file=stan_file)
LOGGER.info(".. done")

for stationid, y in ams.items():
    LOGGER.info(f"Running station {stationid}")
    N = len(y)
    ndone = 0

    for iboot in range(nboot):
        LOGGER.info(f"\tBoot {iboot + 1} / {nboot}")
        rng = np.random.default_rng(SEED)
        yboot = rng.choice(y.values, N)

        censor = np.percentile(yboot, 20)
        dnocens = yboot[yboot >= censor]
        ncens = (yboot < censor).sum()

        sv = sample.StanSamplingVariable(marginal, yboot, censor,
                                         ninits=1)
        stan_data = sv.to_dict()
        marginal.params = {k[1:]: v for k, v
                           in sv.initial_parameters[0].items()}

        # Test shape close to 0 for edge cases
        if iboot < nboot_zero:
            marginal.shape1 = 1e-20
            LOGGER.info("\t\t zero shape")
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
        errmess = f"Error : {marginal}"
        atol = 1e-5
        rtol = 1e-3
        locn = smp.filter(regex="ylocn").values
        assert np.allclose(locn, marginal.locn,
                           atol=atol, rtol=rtol), errmess

        logscale = smp.filter(regex="ylogscale").values
        assert np.allclose(logscale, marginal.logscale,
                           atol=atol, rtol=rtol), errmess

        shape1 = smp.filter(regex="yshape1").values
        assert np.allclose(shape1, marginal.shape1,
                           atol=atol, rtol=rtol), errmess

        # Test data
        i11 = np.array(stan_data["i11"]) - 1
        luncens = smp.filter(regex="luncens").values[i11]
        expected = marginal.logpdf(yboot[i11])
        assert np.allclose(luncens, expected,
                           atol=atol, rtol=rtol), errmess

        cens = smp.filter(regex="^cens").values[i11]
        expected = marginal.cdf(yboot[i11])
        assert np.allclose(cens, expected,
                           atol=atol, rtol=rtol), errmess

        rtol = 1e-2
        atol = 5e-3
        lcens = smp.filter(regex="^lcens").values[i11]
        expected = marginal.logcdf(yboot[i11])
        assert np.allclose(lcens, expected,
                           atol=atol, rtol=rtol), errmess
        atol = 1e-5
        rtol = 1e-3

        lpr = 0.
        for pn in marginals.PARAMETERS:
            lp = smp.filter(regex=f"logprior_{pn}")
            prior = getattr(marginal, f"{pn}_prior")
            expected = prior.logpdf(getattr(marginal, pn))
            assert np.allclose(lp, expected,
                               atol=atol, rtol=rtol), errmess
            lpr += lp.squeeze()

        ll = smp.filter(regex="loglikelihood").values[0]
        expected = marginal.logpdf(dnocens).sum()
        expected += ncens * marginal.logcdf(censor)
        assert np.allclose(ll, expected,
                           atol=atol, rtol=rtol), errmess

        lp = smp.filter(regex="logposterior").values[0]
        expected = -marginal.neglogpost(marginal.params, dnocens,
                                       censor, ncens)
        assert np.allclose(lp, expected,
                           atol=atol, rtol=rtol), errmess

    # Ensures at least 5 simulation beyond 0 shape trials
    assert ndone > nboot_zero + 5

LOGGER.info("Process completed")
