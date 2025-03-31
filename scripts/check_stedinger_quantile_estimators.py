#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-03-31 12:15:21.000583
## Comment : Check Stedinger quantile estimators from
##           Stedinger, J. R. (1983). Design events with specified flood risk.
##           Water Resources Research, 19(2), 511–522.
##           https://doi.org/10.1029/WR019i002p00511
##
## ------------------------------

import sys
import re
import math
from itertools import product as prod
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.stats import t as tstud
from scipy.stats import gamma

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.plot import putils

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
nsamples = 10000
nvals = [5, 50, 100]

aris = np.array([10, 100, 500])
cdf = 1 - 1. / aris

# Prior specifications
mu0 = 10. # prior mean
k0 = 5 # number of prior obs

ks0 = k0 # Assume 5 "prior" obs for lam (same than mu)
lam0 = 1 / 5**2 # prior std equal to 5
a0 = ks0 / 2  # same number of prior obs than mean (could be different)
b0 = ks0 / 2 / lam0 # prior half sum of squares
# i.e. lam ~ gamma.rvs(a=a0, scale=1./b0, size=1000)


# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "check_stedinger_quantiles_estimators"
fout.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

# Sample parameters from prior
lam = gamma.rvs(a=a0, scale=1. / b0, size=nsamples)
mu = norm.rvs(loc=mu0, scale=1 / np.sqrt(lam * k0), size=nsamples)

# Sample data from prior (looping through nval as a way to sample
# repeatdly)
sig = 1. / np.sqrt(lam)
n = 1000
y = np.empty((nsamples, n))
for i in range(n):
    y[:, i] = norm.rvs(loc=mu, scale=sig, size=nsamples)

sc = np.sqrt(b0 * (k0 + 1) / a0 / k0)
Q = tstud.ppf(cdf, loc=mu0, scale=sc, df=2*a0)

for nobs in nvals:
    # Posterior parameters given data y and prior
    yy = y[:, :nobs]
    mub = yy.mean(axis=1)
    S = ((yy-mub[:, None])**2).sum(axis=1)

    mun = (k0 * mu0 + nobs * mub) / (k0 + nobs)
    kn = k0 + nobs
    an = a0 + nobs / 2
    bn = b0 + S / 2 + k0 * nobs * (mub - mu0)**2 / (k0 + nobs) / 2
    lamp = gamma.rvs(a=an, scale=1./bn, size=nsamples)
    mup = norm.rvs(loc=mun, scale=1/np.sqrt(lamp * kn), size=nsamples)

    # Quantiles of predictive posterior distribution
    # See Murphy, K. P. (n.d.). Conjugate Bayesian analysis of the Gaussian distribution.
    sc = np.sqrt(bn * (kn + 1) / an / kn)
    Qp = tstud.ppf(cdf, loc=mun[:, None], scale=sc[:, None], df=2*an)

    # Check posterior param dist
    # i.e. if posterior param when ranked according to prior
    # is uniformly placed
    plt.close("all")
    mosaic = [["mu", "lam"] + ["."] * (len(aris) - 2),
              [f"ari{iari}" for iari in range(len(aris))]]
    ncols = len(mosaic[0])
    nrows = len(mosaic)
    w, h = 5, 4
    fig = plt.figure(figsize=(w * ncols, h * nrows),
                     layout="constrained")
    axs = fig.subplot_mosaic(mosaic)

    for aname, ax in axs.items():
        if aname.startswith("ari"):
            iq = int(re.sub("ari", "", aname))
            ax.hist(Qp[:, iq], bins=200, density=True)
            putils.line(ax, 0, 1, Q[iq], 0, "k--", lw=0.8)

            title = f"Q{aris[iq]}"
            ax.set(title=title, xlabel="Sample", ylabel="freq")
            continue

        elif aname == "mu":
            p = (mu[:, None] - mup[None, :] >= 0).sum(axis=1) / nsamples
            title = "mu"
        elif aname == "lam":
            p = (lam[:, None] - lamp[None, :] >= 0).sum(axis=1) / nsamples
            title = "lam"

        ax.plot(np.sort(p))
        ax.set(title=title, xlabel="Sample", ylabel="U[0,1]")

    plt.show()

    sys.exit()


LOGGER.completed()

