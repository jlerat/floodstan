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

import importlib.util
f = Path(sample.__file__).parent.parent.parent / "tests" /\
    "test_sample_univariate.py"
spec = importlib.util.spec_from_file_location("test_sample_univariate", f)
tsu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsu)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
stationids = tsu.get_stationids()
stationid = stationids[0]

censoring = True

distname = "LogPearson3"

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs"
fout.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = Path(__file__).stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Get data
# ----------------------------------------------------------------------
y = tsu.get_ams(stationid)
censor = y.median() if censoring else np.nanmin(y) - 1.

# Set STAN
stan_nwarm = 10000
stan_nsamples = 5000
stan_nchains = 5

sv = sample.StanSamplingVariable(y, distname, censor,
                                 ninits=stan_nchains)
stan_data = sv.to_dict()
stan_inits = sv.initial_parameters

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

marginal = marginals.factory(distname)
y0 = y.min()
y1 = y.max()

marginal.params_guess(y)
locn, logscale, shape1 = marginal.params
scale = math.exp(logscale)
shape1 = 0.1
locn = math.log(y.quantile(0.98)) + 2 * scale / shape1
marginal.params = [locn, math.log(scale), shape1]

kw = marginal.get_scipy_params()
yy = np.linspace(y0, y1, 1000)
lyy = np.log(yy)
lpdf = pearson3.logpdf(lyy, **kw) - lyy

# Linear continuation
# .. location of reference value
ly = np.log(y)
#dly = (ly.max() - ly.min()) / 20
dly = 1e-4
ly_ref = locn - 2 * scale / shape1 + np.sign(shape1) * dly

y_ref = math.exp(ly_ref)

iout = (lyy - ly_ref) * shape1 < 0

# .. log pdf at the reference value
lpdf_ref = pearson3.logpdf(ly_ref, **kw) - ly_ref

# .. derivative of log pdf at the reference value
alpha = 4. / shape1**2
beta = 2 / shape1
xi = -beta

u_ref = ((ly_ref - locn) / scale - xi) * beta
dlpdf_ref = (((alpha - 1) / u_ref - 1.) * beta / scale) / y_ref

# .. apply continuation
d = yy - y_ref
lpdf2 = np.where(iout, lpdf_ref + dlpdf_ref * d, lpdf)


#ll = pearson3.logpdf(lyy, **kw)
#dll = (ll[2:] - ll[:-2]) / (lyy[2:] - lyy[:-2])
#ll = ll[1:-1]
#lyy = lyy[1:-1]
#iok = np.abs(ll) < 200
#
#u = ((lyy - locn) / scale - xi) * beta
#C = math.log(abs(beta) / gamma_fun(alpha))
#ll2 = C + (alpha - 1) * np.log(u) - u - math.log(scale)
#dll2 = ((alpha - 1) / u  - 1) * beta / scale
#
#plt.close("all")
#fig, ax = plt.subplots(layout="constrained")
#C = math.log(abs(beta) / gamma_fun(alpha))
#ax.plot(lyy[iok], ll[iok], "k-", lw=3)
#ax.plot(lyy[iok], ll2[iok], "-", color="orange")
#tax = ax.twinx()
#tax.plot(lyy[iok], dll[iok], lw=3)
#tax.plot(lyy[iok], dll2[iok], "orange")
#tax.set(ylabel = "diff log likelihood")
#plt.show()


# plot
plt.close("all")
fig, ax = plt.subplots()
ax.plot(yy, lpdf)
ax.plot(yy[iout], lpdf[iout], "r")
ylim = ax.get_ylim()
ax.plot(yy[iout], lpdf2[iout], "--r")
ax.plot(math.exp(ly_ref), lpdf_ref, "or")
ax.plot([y0, y1], [np.nanmax(lpdf)] * 2, "k--")
#ax.set(ylim=ylim)

plt.show()

LOGGER.completed()

