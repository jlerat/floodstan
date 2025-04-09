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
from math import sqrt, log, exp, pi
from itertools import product as prod
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
from scipy.special import erfc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

fout = froot / "outputs" / "check_gamma_temme"
fout.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
def gamma_star(alpha):
    # Coefficients from Temme (1987)
    ak = np.array([5.115471897484e-2, 4.990196893575e-1,
                        9.404953102900e-1, 9.999999625957e-1])
    bk = np.array([1.544892866413e-2, 4.241288251916e-1,
                        8.571609363101e-1, 1.000000000000e+0])
    num, den = 0, 0
    for i in range(4):
        num = num * alpha + ak[3-i]
        den = den * alpha + bk[3-i]

    return num/den

def gamma_logcdf_temme(x, alpha):
    Napprox = 14
    bm = np.zeros(Napprox + 1)
    fm = np.array([-3.33333333e-01,  8.33333333e-02, -1.48148148e-02,
        1.15740741e-03,  3.52733686e-04, -1.78755144e-04,  3.91926318e-05,
       -2.18544851e-06, -1.85406221e-06,  8.29671134e-07, -1.76659527e-07,
        6.70785354e-09,  1.02618098e-08, -4.38203602e-09,  9.14769958e-10])

    # First term of the approximation
    lam = x / alpha
    eta = np.sqrt(2 * (lam - 1. - np.log(lam)))
    eta = np.where(lam < 1, -eta, eta)
    cdf0 = erfc(-eta * np.sqrt(alpha / 2.)) / 2.

    # Approximation coefficients to compute the residuals
    bm[Napprox] = fm[Napprox];
    bm[Napprox - 1] = fm[Napprox - 1];
    for i in range(1, Napprox):
        mb = Napprox-i
        f = fm[mb-1] if mb > 0 else 1.
        bm[mb-1] = f + (mb + 1.) / alpha * bm[mb+1]

    # Compute residual
    S = 0
    for ms in range(1, Napprox+1):
        S += bm[ms-1] * np.power(eta, ms - 1.)

    A = np.exp(-alpha * eta * eta / 2.) / np.sqrt(2. * pi * alpha)
    GS = gamma_star(alpha)
    R = A * S / GS

    # Put it together
    cdf = cdf0 - R
    return np.where(cdf > 0, np.log(cdf), -np.inf)


n = 10
ng, nsigs = n, n
gs = np.logspace(-4, math.log10(2), ng)
sigs = np.logspace(-2, 2, nsigs)

G, S = np.meshgrid(gs, sigs)
E2 = np.zeros_like(G)

y = np.logspace(0, 1, 500)
ly = np.log(y)
m = ly.mean()

def get_params(m, s, g):
    alpha = 4. / g / g
    beta = 2. / g / s
    tau = m - alpha / beta
    return alpha, beta, tau

for idxg, idxs in prod(range(n), range(n)):
    g = G[idxg, idxs]
    s = S[idxg, idxs]
    alpha, beta, tau = get_params(m, s, g)
    if (ly > tau).sum() == 0:
        continue
    x = beta * (ly[ly > tau] - tau)

    # Scipy
    y1 = gamma.logcdf(x, alpha)

    # Temme
    y2 = gamma_logcdf_temme(x, alpha)

    # Error
    E2[idxg, idxs] = np.log10(np.abs(y2 - y1).max())

plt.close("all")
fig, axs = plt.subplots(ncols=2,
                        figsize=(12, 6),
                        layout="constrained")

ax = axs[0]
cnf = ax.contourf(G, S, E2)
E2max = np.nanmax(E2[np.isfinite(E2)])
i1, i2 = np.where(E2 > E2max - 1e-10)
ax.plot(gs[i2], sigs[i1], "ro")

ax.set_xscale("log")
ax.set_yscale("log")
plt.colorbar(cnf)

title = "log10 of max diff in log(gamma cdf) for Temme vs scipy"
ax.set(title=title, xlabel="g", ylabel="s")

ax = axs[1]
g, s = gs[i2], sigs[i1]
alpha, beta, tau = get_params(m, s, g)
x = beta * (ly[ly > tau] - tau)
y1 = gamma.logcdf(x, alpha)
y2 = gamma_logcdf_temme(x, alpha)
ax.plot(x, y1, label="Scipy")
ax.plot(x, y2, label="Temme")
ax.legend(loc=2)
ax.set_xscale("log")

tax = ax.twinx()
tax.plot(x, y2 - y1, "k--", lw=0.8)


plt.show()

LOGGER.completed()

