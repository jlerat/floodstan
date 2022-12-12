import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pytest
import warnings

from cmdstanpy import CmdStanModel

import importlib
from tqdm import tqdm

from nrivfloodfreqstan import gls_spatial, gls_spatial_generate

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent

FIMG = FTESTS / "images"
FIMG.mkdir(exist_ok=True)

TQDM_DISABLE = True

def test_gls_generate():
    # Generate coordinates
    NX = 20
    u = np.linspace(0, 1, NX)
    N = NX*NX
    uu, vv = np.meshgrid(u, u)
    uu, vv = uu.ravel(), vv.ravel()
    w = np.column_stack([uu, vv])

    # Generate predictors
    x = np.column_stack([uu, vv, uu*vv, uu**2, vv**2])
    P = x.shape[1]
    beta = np.random.uniform(-1, 1, size=P)

    mu0 = x.dot(beta)

    # Low noise
    sigma = np.std(mu0)/5
    # Large spatial covariance
    alpha = 3*sigma
    # Long spatial correlation
    rho = 2.

    for kernel in [1, 2]:
        stan_data = {
            "N": N, \
            "P": P, \
            "x": x, \
            "w": w, \
            "beta": beta, \
            "logrho": math.log(rho),
            "logalpha": math.log(alpha), \
            "logsigma": math.log(sigma), \
            "kernel": kernel
        }

        # Clean output folder
        fout = FTESTS / "sampling" / "gls_generate"
        fout.mkdir(parents=True, exist_ok=True)
        for f in fout.glob("*.*"):
            f.unlink()

        # Sample
        nsamples = 16
        smp = gls_spatial_generate.sample(\
                    data=stan_data, \
                    seed=SEED, \
                    iter_warmup=10, \
                    iter_sampling=nsamples, \
                    adapt_engaged=True, \
                    chains=1, \
                    output_dir=fout)
        df = smp.draws_pd()

        y = df.filter(regex="^y", axis=1)

        plt.close("all")
        fig, axs = plt.subplots(ncols=4, nrows=4, \
                            figsize=(12, 12), \
                            layout="tight")
        for i, ax in enumerate(axs.flat):
            if i == 0:
                ys = mu0.reshape((NX, NX))
                title = "mu0"
            else:
                ys = y.iloc[i-1, :].values.reshape((NX, NX))
                title = f"Sample {i}"

            ax.matshow(ys)
            ax.set_title(title)

        fp = FIMG / f"gls_generate_kernel{kernel}.png"
        fig.savefig(fp)



def test_gls():
    # Generate coordinates
    NX = 5
    u = np.linspace(0, 1, NX)
    N = NX*NX
    uu, vv = np.meshgrid(u, u)
    w = np.column_stack([uu.ravel(), vv.ravel()])

    # Generate predictors
    x = np.column_stack([w[:, 0], w[:, 1], np.prod(w, axis=1)])
    P = x.shape[1]
    beta = np.random.uniform(-1, 1, size=P)

    mu0 = x.dot(beta)
    sigma = np.std(mu0)/5
    alpha = 3*sigma
    rho = 2.

    stan_data = {
        "N": N, \
        "P": P, \
        "x": x, \
        "w": w, \
        "beta": beta, \
        "logrho": math.log(rho),
        "logalpha": math.log(alpha), \
        "logsigma": math.log(sigma), \
        "kernel": 1
    }

    # Clean output folder
    fout = FTESTS / "sampling" / "gls"
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Generate
    nsamples = 1
    smp = gls_spatial_generate.sample(\
                data=stan_data, \
                seed=SEED, \
                iter_warmup=10, \
                iter_sampling=nsamples, \
                adapt_engaged=True, \
                chains=1, \
                output_dir=fout)
    df = smp.draws_pd().filter(regex="^y", axis=1)

    # Select points
    ipts = np.random.choice(np.arange(N), 2*NX, replace=False)

    # sample
    theta_prior = np.row_stack([np.zeros(P), 5*np.ones(P)])

    stan_data = {
        "N": len(ipts), \
        "P": P, \
        "x": x[ipts], \
        "w": w[ipts], \
        "y": df.values[0, ipts], \
        "kernel": 1, \
        "logrho_prior": [math.log(0.5), 0.5], \
        "logalpha_prior": [0., 2], \
        "logsigma_prior": [0., 2], \
        "theta_prior": theta_prior
    }
    smp = gls_spatial.sample(\
                data=stan_data, \
                seed=SEED, \
                iter_warmup=5000, \
                iter_sampling=10000, \
                chains=5, \
                output_dir=fout)
    df = smp.draws_pd()


    import pdb; pdb.set_trace()

