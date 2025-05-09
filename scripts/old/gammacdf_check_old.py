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
from scipy.stats import lognorm
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


def gammastar(alpha):
    num = 0.
    den = 0.

    ak = np.zeros(4)
    bk = np.zeros(4)

    ak[0] = 5.115471897484e-2
    bk[0] = 1.544892866413e-2
    ak[1] = 4.990196893575e-1
    bk[1] = 4.241288251916e-1
    ak[2] = 9.404953102900e-1
    bk[2] = 8.571609363101e-1
    ak[3] = 9.999999625957e-1
    bk[3] = 1.000000000000e+0

    for i in range(4):
        num = num * alpha + ak[3 - i];
        den = den * alpha + bk[3 - i];

    return num/den

def gamma_cdf(x, alpha):
    fm = np.array([-3.33333333e-01,  8.33333333e-02, -1.48148148e-02,
        1.15740741e-03,  3.52733686e-04, -1.78755144e-04,  3.91926318e-05,
       -2.18544851e-06, -1.85406221e-06,  8.29671134e-07, -1.76659527e-07,
        6.70785354e-09,  1.02618098e-08, -4.38203602e-09,  9.14769958e-10 ])

    Napprox = len(fm) - 1;
    bm = np.zeros(Napprox + 1)

    # First term of the approximation
    lam = x / alpha
    if lam <= 0:
        return np.nan
    eta = math.sqrt(2 * (lam - 1. -math.log(lam)))
    eta = -eta if lam < 1 else eta
    cdf0 = erfc(-eta * math.sqrt(alpha / 2.)) / 2.

    # Approximation coefficients to compute the residual
    bm[Napprox] = fm[Napprox]
    bm[Napprox - 1] = fm[Napprox - 1]
    f = 0.
    for i in range(1, Napprox):
        mb = Napprox - i
        f = fm[mb - 1] if mb > 0 else 1.
        bm[mb - 1] = f + (mb + 1.) / alpha * bm[mb + 1]

    # Compute residual
    S = 0
    for ms in range(1, Napprox + 1):
        S += bm[ms - 1] * math.pow(eta, ms - 1)

    A = math.exp(-alpha * eta * eta / 2.) / math.sqrt(2. * math.pi * alpha)
    GS = gammastar(alpha)
    R = A * S / GS

    # Put it together
    cdf = cdf0 - R

    return cdf

ng = 10
gs = np.logspace(-2, math.log10(2), ng)
nsigs = 10
sigs = np.logspace(-2, 2, nsigs)

worked = 0
for ic, (g, s) in enumerate(prod(gs, sigs)):
    alpha = 4. / g / g
    beta = 2. / g / s
    abs_beta = abs(beta)

    x = np.logspace(0, 3, 500)
    m = x.mean()

    tau = m - alpha / beta
    lx = beta * (np.log(x) - tau)

    y0 = lognorm.cdf(x, scale=math.exp(m), s=s)
    y1 = gamma.cdf(lx, alpha, scale=abs_beta)

    if (y1 > 0).sum() == 0:
        txt = f"[{ic+1:2d}/{ng*nsigs}] No cdf > 0 for "\
              + f"g={g:2.2e} (a={alpha:2.2e}) s={s:2.2e}"
        LOGGER.info(txt)
        continue

    worked += 1

    if alpha < 100:
        plt.plot(x, y0)
        plt.plot(x, y1)
        plt.show()
        sys.exit()


    y2 = np.array([gamma_cdf(lxx, alpha) for lxx in lx])

    stan_data = {
        "x": lx,
        "N": len(x),
        "a": alpha
    }
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
    y3 = smp.filter(regex="cdf").values

    isfin = (y1 > 0) & (y2 > 0) & (y3 > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l1 = np.log(y1)[isfin]
        l2 = np.log(y2)[isfin]
        l3 = np.log(y3)[isfin]

    #assert np.allclose(l1, l2, atol=1e-5)
    assert np.allclose(l1, l3, atol=1e-5)

LOGGER.info(f"Valid configurations = {worked} / {ng * nsigs}")

LOGGER.completed()

