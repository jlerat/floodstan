import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import solve

KERNEL_CODES = {
    1: "Gaussian",
    2: "Exponential"
    }

# Path to priors
FPRIORS = Path(__file__).resolve().parent / "priors"


def kernel_covariance(w, rho, alpha, sigma, kernel):
    N = len(w)
    d = squareform(pdist(w)) / rho

    if kernel == 1:
        K = np.exp(-d * d / 2)

    elif kernel == 2:
        K = np.exp(-d)

    else:
        txt = "/".join(list(KERNEL_CODES))
        errmsg = f"Expected kernel in {txt}, got {kernel}."
        raise ValueError(errmsg)

    # Add alpha and sigma error
    return alpha * alpha * K + sigma**2 * np.eye(N)


def kernel_sqroot(K, rcond=1e-10):
    U, S, Vt = np.linalg.svd(K)

    Smax = S.max()
    Smin = Smax * rcond
    S = np.maximum(S, Smin)
    Sq = np.diag(np.sqrt(S))
    L = U @ Sq @ Vt
    return L


def prepare(x, w, y,
            logrho_prior,
            logalpha_prior,
            logsigma_prior,
            theta_prior=None,
            logrho_lower=-10,
            logrho_upper=20,
            kernel=1):
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
    if x.ndim != 2:
        raise ValueError("Expected 2d array for x.")

    if w.ndim != 2:
        raise ValueError("Expected 2d array for w.")

    if w.shape[1] != 2:
        raise ValueError("Expected 2nd dim of w to be of length 2.")

    if y.ndim != 1:
        raise ValueError("Expected 1d array for y.")

    N = len(y)
    P = x.shape[1]
    if x.shape[0] != N:
        raise ValueError(f"Expected {N} points in x, got {x.shape[0]}.")

    if w.shape[0] != N:
        raise ValueError(f"Expected {N} points in w, got {w.shape[0]}.")

    # theta priors
    if theta_prior is None:
        theta_prior = np.row_stack([np.zeros(P), 3 * np.ones(P)])

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
        "kernel": kernel,
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
    if x.ndim != 2:
        raise ValueError("Expected 2d array for x.")

    N = len(x)
    Q, R = np.linalg.qr(x)

    # Set diagonal of R positive
    pos = np.diag((np.diag(R) > 0).astype(int) * 2 - 1)
    R = pos.dot(R)
    Q = Q.dot(pos)

    # Standardise as per stan QR factorisation
    # See https://mc-stan.org/docs/stan-users-guide/QR-reparameterization.html
    Q_ast = Q * math.sqrt(N-1)
    R_ast = R / math.sqrt(N-1)
    return Q_ast, R_ast


def generate(stan_data, params, conditional=True):
    # Get data
    N = stan_data["N"]
    x = stan_data["x"]
    w = stan_data["w"]
    y = stan_data["y"]
    ivalid = stan_data["ivalid"]-1
    kernel = stan_data["kernel"]

    # Check params
    for pn in ["alpha", "rho", "sigma"]:
        lpn = f"log{pn}"
        if lpn not in params.columns:
            errmess = f"Expected a column {lpn} in params."
            raise ValueError(errmess)

    cnbeta = [cn for cn in params.columns if re.search("beta", cn)]
    if len(cnbeta) == 0:
        errmess = "Cannot find columns 'beta' in params."
        raise ValueError(errmess)

    # generate data
    M = len(params)
    eps = np.random.normal(size=(M, N))
    samples = np.nan * np.zeros([M, N])
    for i, (_, theta) in enumerate(params.iterrows()):
        beta = theta.loc[cnbeta]
        rho = math.exp(theta.logrho)
        alpha = math.exp(theta.logalpha)
        sigma = math.exp(theta.logsigma)

        # Derived parameters
        mu = x.dot(beta)
        try:
            K = kernel_covariance(w, rho, alpha, sigma, kernel)
            L = kernel_sqroot(K)
        except Exception:
            continue

        raw = mu + L.dot(eps[i])

        if conditional:
            K22 = K[ivalid[:, None], ivalid[None, :]]
            u = y[ivalid] - raw[ivalid]
            try:
                v = solve(K22, u, assume_a="pos")
            except Exception:
                v = solve(K22, u)

            K12 = K[:, ivalid]
            ys = raw + K12 @ v
        else:
            ys = raw

        samples[i, :] = ys

    return samples
