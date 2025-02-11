import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.stats import ttest_1samp

import pytest
import warnings

from hydrodiy.plot import putils

import importlib
from tqdm import tqdm

from floodstan import marginals, sample, copulas
from floodstan import univariate_censored_sampling

from test_sample_univariate import get_stationids
from test_sample_univariate import get_ams


SEED = 5446

FTESTS = Path(__file__).resolve().parent

@pytest.mark.cook
@pytest.mark.parametrize("marginal",
                         list(sample.MARGINAL_NAMES.keys()))
@pytest.mark.parametrize("stationid",
                         get_stationids()[:3])
def test_univariate_sampling(marginal, stationid, allclose):
    # Testing univariate sampling following the process described by
    # Samantha R Cook, Andrew Gelman & Donald B Rubin (2006)
    # Validation of Software for Bayesian Models Using Posterior Quantiles,
    # Journal of Computational and Graphical Statistics, 15:3, 675-692,
    # DOI: 10.1198/106186006X136976

    nchains = 5

    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 50
    nrepeat = 100


    nrows, ncols = 3, 3
    axwidth, axheight = 5, 5
    print("\n")

    y = get_ams(stationid)
    N = len(y)

    # Setup image
    plt.close("all")
    w, h = axwidth*ncols, axheight*nrows
    fig, ax = plt.subplots(figsize=(w, h), layout="tight")

    if marginal in ["Gumbel", "LogNormal", "Normal", "Gamma"]:
        parnames = ["locn", "logscale"]
    else:
        parnames = ["locn", "logscale", "shape1"]

    dist = marginals.factory(marginal)
    dist.params_guess(y)

    if dist.shape1<marginals.SHAPE1_LOWER \
            or dist.shape1>marginals.SHAPE1_UPPER:
        pytest.skip("Shape parameter outside of accepted bounds.")

    # Prior distribution centered around dist params
    ylocn_prior = [dist.locn, abs(dist.locn)*0.5]
    ylogscale_prior = [dist.logscale, abs(dist.logscale)*0.5]
    yshape1_prior = [max(0.1, dist.shape1), 0.2]

    test_stat = []
    # .. double the number of tries to get nrepeat samples
    # .. in case of failure
    for repeat in range(2*nrepeat):
        desc = f"[{stationid}] Testing uniform sampling for "+\
                    f"marginal {marginal} ({repeat+1}/{nrepeat})"
        print(desc)

        # Generate parameters from prior
        try:
            dist.locn = norm.rvs(*ylocn_prior)
            dist.logscale = norm.rvs(*ylogscale_prior)
            dist.shape1 = norm.rvs(*yshape1_prior)
        except:
            continue

        # Generate data from prior params
        ysmp = dist.rvs(N)

        # Configure stan data and initialisation
        try:
            sv = sample.StanSamplingVariable(ysmp, marginal,
                                             ninits=nchains)
        except:
            continue

        sv.locn_prior = ylocn_prior
        sv.logscale_prior = ylogscale_prior
        sv.shape1_prior = yshape1_prior

        stan_data = sv.to_dict()
        stan_inits = sv.initial_parameters

        # Clean output folder
        fout = FTESTS / "sampling" / "univariate" / stationid / marginal
        fout.mkdir(parents=True, exist_ok=True)
        for f in fout.glob("*.*"):
            f.unlink()

        # Sample
        try:
            smp = univariate_censored_sampling(data=stan_data,
                                               inits=stan_inits,
                                               chains=nchains,
                                               seed=SEED,
                                               iter_warmup=5000,
                                               iter_sampling=500,
                                               output_dir=fout)
        except Exception as err:
            continue

        # Get sample data
        df = smp.draws_pd()

        # Test statistic
        Nsmp = len(df)
        test_stat.append([(df.loc[:, f"y{cn}"]<dist[cn]).sum()/Nsmp\
                for cn in parnames])

        # Stop iterating when the number of samples is met
        if repeat>nrepeat-2:
            break

    test_stat = pd.DataFrame(test_stat, columns=parnames)

    putils.ecdfplot(ax, test_stat)
    ax.axline((0, 0), slope=1, linestyle="--", lw=0.9)

    title = f"{marginal} - {len(test_stat)} samples / {nrepeat}"
    ax.set_title(title)

    fig.suptitle(f"Station {stationid}")
    fp = FTESTS / "images" / f"univariate_sampling_{stationid}_{marginal}.png"
    fp.parent.mkdir(exist_ok=True)
    fig.savefig(fp)

