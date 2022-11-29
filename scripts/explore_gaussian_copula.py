#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2022-11-28 16:33:54.641687
## Comment : test approximation to compute gaussian copula
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path

from itertools import product as prod

#import warnings
#warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import mvn
from scipy.integrate import nquad


import matplotlib.pyplot as plt


from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils

from tqdm import tqdm

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

#  see  Zvi Drezner & G. O. Wesolowsky (1990) On the computation of the
#  bivariate normal integral,
#  Journal of Statistical Computation and Simulation, 35:1-2, 101-107, DOI: 10.1080/00949659008811236
#  https://www.tandfonline.com/doi/pdf/10.1080/00949659008811236?needAccess=true
#
# FUNCTION BV(H 1 ,H2,R)
# DIMENSION X(5),W(5)
# DATA X/.04691008,.23076534,.5,.76923466,.95308992/
# DATA W/.O18854042,.038088059,.0452707394,.038088059,.018854042/
# H12 =(Hl*Hl+ H2*H2)/2.
# H3 = H1 *H2
# BV = 0.
# DO 1 I= 1,5
# RR = R*X(I)
# RR2= 1.-RR*RR
# 1 BV = BV + W(I)*EXP((RR*H3 - H12)/RR2)/SQRT(RR2)
# BV = BV*R + GAUSS(H 1)*GAUSS(H2)
# RETURN
# END

def cdf1(h1, h2, r):
    m = -r*norm.pdf(h1)/norm.cdf(h1)
    s2 = 1+r*h1*m-m**2
    return norm.cdf(h1)*norm.cdf((h2-m)/math.sqrt(s2))

def cdf2(x, y, r):
    a = [0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992]
    b = [0.018854042, 0.038088059, 0.0452707394, 0.038088059, 0.018854042]

    h12 =(x*x+ y*y)/2.
    h3 = x*y
    bv = 0.
    for i in range(5):
        rr = r*a[i]
        rr2 = 1.-rr*rr
        bv += b[i]*math.exp((rr*h3 - h12)/rr2)/math.sqrt(rr2)

    bv = bv*r + norm.cdf(x)*norm.cdf(y)

    return bv


nsamples = 10000000
mu = np.zeros(2)
Sigma = np.eye(2)
qq = 0.2*np.arange(1, 5)
censors = norm.ppf(qq)
rhos = np.linspace(0., 0.99, 5)

nc = len(censors)
total = len(rhos)*(nc*(nc+1)//2)
tbar = tqdm(desc="Processing", total=total)
res = []


for rho in rhos:
    Sigma[0, 1] = rho
    Sigma[1, 0] = rho
    h = np.random.multivariate_normal(mean=mu, cov=Sigma, size=nsamples)

    for c1, c2 in prod(censors, censors):
        if c2>c1:
            continue

        tbar.update()
        ccn1 = cdf1(c1, c2, rho)
        ccn2 = cdf2(c1, c2, rho)

        ii = (h[:, 0]<=c1) & (h[:, 1]<=c2)
        ccs = np.sum(ii)/nsamples

        # Using scipy utility
        lower = np.zeros(2) # Does not matter here
        upper = np.array([c1, c2])
        infin = np.zeros(2)
        correl = np.array([rho])
        err, csp, info = mvn.mvndst(lower, upper, infin, correl)

        # Numerical integration
        def fun(x, y):
            z = x*x-2*rho*x*y+y*y
            r2 = 1-rho*rho
            return math.exp(-z/r2/2)/2/math.pi/math.sqrt(r2)

        csa, err = nquad(fun, [[-np.inf, c1], [-np.inf, c2]])

        if abs(ccs-csa)>0.1:
            sys.exit()


        # Store
        dd = {\
            "c1": c1, \
            "c2": c2, \
            "rho": rho, \
            "cdf_approx1": ccn1, \
            "cdf_approx2": ccn2, \
            "cdf_approx3": csp, \
            "cdf_approx4": ccs, \
            "cdf_integ": csa
        }

        res.append(dd)

res = pd.DataFrame(res)


plt.close("all")
names = ["Mee 1983", "Drezner 1990", "Genz 2000", "Sampling"]
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), layout="tight")
for iax, ax in enumerate(axs.flat):
    for rho in rhos:
        ii = res.rho == rho
        x = res.cdf_integ[ii]
        y = res.loc[ii, f"cdf_approx{iax+1}"]
        ax.plot(x, y, "o", \
                    label=f"$\\rho=${rho:0.2f}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.9)
    ax.legend(loc=4)
    ax.set_title(f"Approx: {names[iax]}")

fp = froot / f"{basename}.png"
fig.savefig(fp)


LOGGER.info("Process completed")

