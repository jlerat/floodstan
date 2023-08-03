import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import pytest
import warnings

from cmdstanpy import CmdStanModel

import importlib
from tqdm import tqdm

from nrivfloodfreqstan import gls_spatial, gls_spatial_generate, test_glsfun
from nrivfloodfreqstan import gls, sample

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent

FIMG = FTESTS / "images"
FIMG.mkdir(exist_ok=True)

TQDM_DISABLE = True

def generate_data(NX):
    u = np.linspace(0, 1, NX)
    N = NX*NX
    uu, vv = np.meshgrid(u, u)
    uu, vv = uu.ravel(), vv.ravel()
    w = np.column_stack([uu, vv])
    x = np.column_stack([uu, vv, uu*vv, uu**2, vv**2])
    y = uu
    P = x.shape[1]
    return x, w, y, N, P, NX


def test_gls_prepare(allclose):
    x, w, y, N, P, NX = generate_data(20)
    stan_data = gls.prepare(x, w, y, \
                    logrho_prior=[0, 1], \
                    logalpha_prior=[0, 2], \
                    logsigma_prior=[0, 2])
    assert stan_data["N"] == N
    assert stan_data["P"] == P
    assert stan_data["Nvalid"] == N
    assert stan_data["theta_prior"].shape == (2, P)

    y[:3] = np.nan
    stan_data = gls.prepare(x, w, y, \
                    logrho_prior=[0, 1], \
                    logalpha_prior=[0, 2], \
                    logsigma_prior=[0, 2])
    assert stan_data["N"] == N
    assert stan_data["Nvalid"] == N-3
    assert allclose(stan_data["ivalid"], np.arange(4, N+1))

    y[3] = np.inf
    stan_data = gls.prepare(x, w, y, \
                    logrho_prior=[0, 1], \
                    logalpha_prior=[0, 2], \
                    logsigma_prior=[0, 2])
    assert stan_data["N"] == N
    assert stan_data["Nvalid"] == N-4
    assert allclose(stan_data["ivalid"], np.arange(5, N+1))


def test_kernel(allclose):
    x, w, y, N, P, NX = generate_data(5)
    beta = np.random.uniform(-1, 1, size=P)
    rho = 1
    alpha = 2
    sigma = 3
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    for kernel in ["Gaussian", "Exponential"]:
        K = gls.kernel_covariance(w, rho, alpha, sigma, kernel)

        # Basic tests
        assert K.shape == (N, N)
        assert allclose(np.diag(K), alpha**2+sigma**2)

        # Tests against stan
        stan_data = {
            "N": N, \
            "P": P, \
            "x": x, \
            "w": w, \
            "beta": beta, \
            "logrho": math.log(rho), \
            "logalpha": math.log(alpha), \
            "logsigma": math.log(sigma), \
            "kernel": gls.KERNEL_CODES[kernel]
        }
        smp = test_glsfun.sample(data=stan_data, \
                            chains=1, iter_warmup=0, iter_sampling=1, \
                            fixed_param=True, show_progress=False)
        smp = smp.draws_pd().squeeze()

        Ks = smp.filter(regex="^K").values.reshape((N, N))
        assert allclose(K, Ks, rtol=0, atol=1e-5)

        Ls = smp.filter(regex="^L").values.reshape((N, N))
        L = np.linalg.cholesky(K).T
        assert allclose(L, Ls, rtol=0, atol=1e-5)


def test_QR(allclose):
    x, w, y, N, P, NX = generate_data(5)
    beta = np.random.uniform(-1, 1, size=P)
    rho = 1
    alpha = 2
    sigma = 3
    kernel = "Gaussian"
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    # Compute QR decomposition
    Q_ast, R_ast = gls.get_QR_matrices(x)
    assert allclose(Q_ast.dot(R_ast), x)
    d = Q_ast.T.dot(Q_ast)
    assert allclose(np.diag(d), N-1)
    assert allclose(d[np.triu_indices(P, 1)], 0.)

    # Tests against stan
    stan_data = {
        "N": N, \
        "P": P, \
        "x": x, \
        "w": w, \
        "beta": beta, \
        "logrho": math.log(rho), \
        "logalpha": math.log(alpha), \
        "logsigma": math.log(sigma), \
        "kernel": gls.KERNEL_CODES[kernel]
    }
    smp = test_glsfun.sample(data=stan_data, \
                        chains=1, iter_warmup=0, iter_sampling=1, \
                        fixed_param=True, show_progress=False)
    smp = smp.draws_pd().squeeze()

    Q_ast_s = np.column_stack(np.array_split(smp.filter(regex="^Q_ast"), P))
    assert allclose(Q_ast, Q_ast_s, rtol=0, atol=1e-5)

    R_ast_s = np.column_stack(np.array_split(smp.filter(regex="^R_ast"), P))
    assert allclose(R_ast, R_ast_s, rtol=0, atol=1e-5)
    assert allclose(Q_ast_s.dot(R_ast_s), x, atol=1e-5, rtol=0.)


def test_gls_generate_stan():
    x, w, y, N, P, NX = generate_data(20)
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

        fp = FIMG / f"gls_generate_stan_kernel{kernel}.png"
        fig.savefig(fp)


def test_gls_generate(allclose):
    x, w, y, N, P, NX = generate_data(30)

    sigma, rho = np.meshgrid([0.1, 1, 2, 3], [0.01, 0.1, 0.2, 0.4])
    sigma, rho = sigma.ravel(), rho.ravel()
    M = len(sigma)
    beta = np.repeat(np.random.choice([-1, 0, 1], size=(P))[None, :], M, 0)
    alpha = 3*np.ones_like(sigma)

    # Coordinates
    u = np.linspace(0, 1, NX)
    uu, vv = np.meshgrid(u, u)

    # Random obs
    y = x.dot(beta[0])
    k = np.random.choice(np.arange(N), N-20, replace=False)
    y[k] = np.nan
    ivalid = np.where(pd.notnull(y))[0]+1
    stan_data = {"N": N, "x": x, "w": w, "y": y, \
                    "ivalid": ivalid, "kernel": 1}

    smps = pd.DataFrame(np.column_stack([beta, \
                            np.log(sigma), \
                            np.log(rho), \
                            np.log(alpha)]))
    smps.columns = [f"beta_{i}" for i in range(1, P+1)]\
                    +["logsigma", "logrho", "logalpha"]

    for conditional in [True, False]:
        ys = gls.generate(stan_data, smps, conditional)

        if conditional:
            err = ys[:, ivalid-1]-y[ivalid-1][None, :]
            assert allclose(np.abs(err), 0.)


        plt.close("all")
        fig, axs = plt.subplots(ncols=4, nrows=4, \
                            figsize=(15, 15), \
                            layout="tight")
        for i, ax in enumerate(axs.flat):
            rho = math.exp(smps.iloc[i].logrho)
            alpha = math.exp(smps.iloc[i].logalpha)
            sigma = math.exp(smps.iloc[i].logsigma)
            params = f"R:{rho:0.1f} A:{alpha:0.1f} S:{sigma:0.1f}"
            txt = ax.text(0.03, 0.97, params, transform=ax.transAxes, \
                        va="top", ha="left", color="k", fontsize=16)
            txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='w'),
                           path_effects.Normal()])

            yys = ys[i].reshape((NX, NX))
            ax.contourf(uu, vv, yys)


            if conditional:
                ax.plot(*w[ivalid-1].T, "k+")

        fp = FIMG / f"gls_generate_cond{conditional}.png"
        fig.savefig(fp)



def test_gls_sample():
    # Generate coordinates
    x, w, _, N, P, NX = generate_data(10)
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
    smps = gls_spatial_generate.sample(\
                data=stan_data, \
                seed=SEED, \
                iter_warmup=10, \
                iter_sampling=nsamples, \
                adapt_engaged=True, \
                chains=1, \
                output_dir=fout)
    smp = smps.draws_pd().iloc[0]

    # Select points
    yfull = smp.filter(regex="^y").values
    ipts = np.random.choice(np.arange(N), N-NX, replace=False)
    y = yfull.copy()
    y[ipts] = np.nan

    # sample
    stan_data = gls.prepare(x, w, y, \
                    logrho_prior=[0, 3], \
                    logalpha_prior=[0, 6], \
                    logsigma_prior=[0, 6])

    smp = gls_spatial.sample(\
                data=stan_data, \
                seed=SEED, \
                iter_warmup=10000, \
                iter_sampling=10000, \
                chains=5, \
                output_dir=fout)
    df = smp.draws_pd()

    # Plot
    plt.close("all")
    fig, axs = plt.subplots(ncols=4, nrows=4, \
                        figsize=(12, 12), \
                        layout="tight")
    for i, ax in enumerate(axs.flat):
        if i == 0:
            ys = yfull.reshape((NX, NX))
            title = "data unmasked"
        else:
            dd = df.iloc[i]
            beta = dd.filter(regex="beta").values
            logrho = dd.logrho
            logalpha = dd.logalpha
            logsigma = dd.logsigma

            stan_data = {
                    "N": N, \
                    "P": P, \
                    "x": x, \
                    "w": w, \
                    "beta": beta, \
                    "logrho": logrho, \
                    "logalpha": logalpha, \
                    "logsigma": logsigma, \
                    "kernel": 1
            }
            smps = gls_spatial_generate.sample(\
                data=stan_data, \
                seed=SEED, \
                iter_warmup=10, \
                iter_sampling=1, \
                adapt_engaged=True, \
                chains=1, \
                output_dir=fout)
            smp = smps.draws_pd().iloc[0]
            ys = smp.filter(regex="^y").values.reshape((NX, NX))
            title = f"Sample {i}"

        ax.matshow(ys)
        ax.set_title(title)

    fp = FIMG / f"gls_sample.png"
    fig.savefig(fp)


