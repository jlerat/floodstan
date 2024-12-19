import sys, re, math, json
from pathlib import Path
import logging
import numbers

from datetime import datetime
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

KERNEL_CODES = {
    "Gaussian": 1,
    "Exponential": 2
    }
KERNEL_CODES_INV = {code:name for name, code in KERNEL_CODES.items()}


# BOUNDS

# Path to priors
FPRIORS = Path(__file__).resolve().parent / "priors"


def kernel_covariance(w, rho, alpha, sigma, kernel):
    N = len(w)
    d = squareform(pdist(w))/rho
    if kernel == "Gaussian":
        K = np.exp(-d*d/2)
    elif kernel == "Exponential":
        K = np.exp(-d)
    else:
        txt = "/".join(list(KERNEL_CODES))
        errmsg = f"Expected kernel in {txt}, got {kernel}."
        raise ValueError(errmsg)

    # Add alpha and sigma error
    return alpha*alpha*K+sigma**2*np.eye(N)


def prepare(x, w, y,
            logrho_prior,
            logalpha_prior,
            logsigma_prior,
            theta_prior=None,
            logrho_lower=-10,
            logrho_upper=20,
            kernel="Gaussian"):
    """ Prepare stan data for GLS model sampling.

    Parameters
    ----------
    x : numpy.ndarray
        Predictors. Array [NxP].
    w : numpy.ndarray
        Spatial coordinates. Array [Nx2].
    y : numpy.ndarray
        Predictand. Array [Nx1]
    logrho_prior: list
        Definition of normal prior (mean, std) for spatial
        correlation parameter.
    logalpha_prior: list
        Definition of normal prior (mean, std) for spatial
        correlation parameter.
    logsigma_prior: list
        Definition of normal prior (mean, std) for error
        standard deviation.
    theta_prior : numpy.ndarray
        Definition of normal priors for each parameters.
        Array [Px2] (mean and std for each parameter).
    kernel : str
        Kernel choice. Either Gaussian or Exponential.
    """

    # Check inputs
    x = np.array(x).astype(np.float64)
    w = np.array(w).astype(np.float64)
    y = np.array(y).astype(np.float64)
    assert x.ndim==2, "Expected 2d array for x."
    assert w.ndim==2, "Expected 2d array for w."
    assert w.shape[1] == 2, "Expected 2nd dim of w to be of length 2."
    assert y.ndim==1, "Expected 1d array for y."

    N = len(y)
    P = x.shape[1]
    assert x.shape[0] == N, f"Expected {N} points in x, got {x.shape[0]}."
    assert w.shape[0] == N, f"Expected {N} points in w, got {w.shape[0]}."

    # theta priors
    if theta_prior is None:
        theta_prior = np.row_stack([np.zeros(P), 3*np.ones(P)])

    # indexes
    valid = pd.notnull(y) & np.isfinite(y)
    ivalid = np.where(valid)[0]+1
    Nvalid = len(ivalid)

    # Create data dict
    stan_data = {
        "N": N,
        "P": P,
        "Nvalid": Nvalid,
        "x": x,
        "w": w,
        "y": y,
        "ivalid": ivalid,
        "kernel": KERNEL_CODES[kernel],
        "logrho_prior": logrho_prior,
        "logalpha_prior": logalpha_prior,
        "logsigma_prior": logsigma_prior,
        "logrho_lower": logrho_lower,
        "logrho_upper": logrho_upper,
        "theta_prior": theta_prior
    }

    # Initial values set to prior mean
    stan_inits = {
        "logrho": logrho_prior[0],
        "logalpha": logalpha_prior[0],
        "logsigma": logsigma_prior[0],
        "theta": theta_prior[0]
    }

    return stan_data, stan_inits


def get_QR_matrices(x):
    x = np.array(x).astype(np.float64)
    assert x.ndim==2, "Expected 2d array for x."
    N = len(x)
    Q, R = np.linalg.qr(x)

    # Set diagonal of R positive
    pos = np.diag((np.diag(R)>0).astype(int)*2-1)
    R = pos.dot(R)
    Q = Q.dot(pos)

    # Standardise as per stan QR factorisation
    # See https://mc-stan.org/docs/stan-users-guide/QR-reparameterization.html
    Q_ast = Q*math.sqrt(N-1)
    R_ast = R/math.sqrt(N-1)
    return Q_ast, R_ast


def generate(stan_data, samples, conditional=False):
    # Get data
    N = stan_data["N"]
    x = stan_data["x"]
    w = stan_data["w"]
    y = stan_data["y"]
    ivalid = stan_data["ivalid"]-1
    kernel = KERNEL_CODES_INV[stan_data["kernel"]]

    # generate data
    M = len(samples)
    eps = np.random.normal(size=(M, N))
    samples_generated = np.zeros([M, N])
    for i, (_, smp) in enumerate(samples.iterrows()):
        beta = smp.filter(regex="beta").values
        rho = math.exp(smp.logrho)
        alpha = math.exp(smp.logalpha)
        sigma = math.exp(smp.logsigma)

        # Derived parameters
        mu = x.dot(beta)
        Sigma = kernel_covariance(w, rho, alpha, sigma, kernel)
        L = np.linalg.cholesky(Sigma)
        raw = mu+L.dot(eps[i])

        if conditional:
            Sigma22inv = np.linalg.inv(Sigma[ivalid[:, None], ivalid[None, :]])
            Sigma12 = Sigma[:, ivalid]
            ys = raw+Sigma12.dot(Sigma22inv).dot(y[ivalid]-raw[ivalid])
        else:
            ys = raw

        samples_generated[i, :] = ys

    return samples_generated
