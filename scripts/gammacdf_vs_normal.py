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
from scipy.special import erf, gammainc
from scipy.special import gamma as gammafun

import matplotlib.pyplot as plt

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

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
mu = 3
sig = 2
x = np.linspace(mu-5*sig, mu+5*sig, 100)

y1 = norm.cdf(x, loc=mu, scale=sig)

u = (x - mu) / sig
y2 = (1 + np.sign(u) * gamma.cdf(u**2/2, a=0.5) * gammafun(0.5) / math.sqrt(math.pi)) / 2


g = 0.05
alpha = 4 / g / g
beta = 2 / g / sig
tau = mu - 2 * sig / g
v = (x - tau) * beta
y3 = gamma.cdf(v, a=alpha)

lam = v / alpha
lamp1 = lam + 1
C0 = 1/ lamp1
C1 = -lam / lamp1**3
C2 = lam * (2*lam - 1) / lamp1**5
C3 = -lam * (6*lam**2 - 8 * lam + 1) / lamp1**7
C4 = lam * (24*lam**3 - 58 * lam**2 + 22 * lam - 1) / lamp1**9
sys.exit()


#X = np.column_stack(

plt.close("all")
fig, ax = plt.subplots()
ax.plot(x, y1, lw=4)
ax.plot(x, y2, lw=0.6)
ax.plot(x, y3, lw=0.6)
txt = f"mu={mu:0.2f}\nsig={sig:0.2f}\ng={g:0.2f}\n\n"\
      + f"tau={tau:0.2f}\nalpha={alpha:0.3f}\nbeta={beta:0.3f}"
ax.text(0.01, 0.99, txt, transform=ax.transAxes,
        va="top", ha="left")
putils.line(ax, 0, 1, mu, 0, "k--", lw=0.8)

tax = ax.twinx()
tax.plot(x, y3 - y1, "k-", lw=0.5)
putils.line(tax, 1, 0, 0, 0, "k-", lw=0.8)

plt.show()

LOGGER.completed()

