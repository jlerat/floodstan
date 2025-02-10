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

from floodstan import marginals
from floodstan import stan_test_marginal, stan_test_copula

from floodstan import report, sample
from floodstan import bivariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent

STATIONIDS = get_stationids()

@pytest.mark.parametrize("copula", sample.COPULA_NAMES_STAN)
@pytest.mark.parametrize("stationid", STATIONIDS)
@pytest.mark.parametrize("censoring", [False, True])
def test_bivariate_sampling_satisfactory(copula, stationid, censoring, allclose):
    LOGGER = sample.get_logger(stan_logger=False)

    stan_nwarm = 5000
    stan_nsamples = 1000
    stan_nchains = 5

    y = get_ams(stationid)
    sids = STATIONIDS.copy()
    sids.remove(stationid)
    stationid2 = np.random.choice(sids)
    z = get_ams(stationid2)
    N = len(y)

    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    censor = y.median() if censoring else -10
    yv = sample.StanSamplingVariable(y, "GEV", censor,
                                     ninits=stan_nchains)
    censor = z.median() if censoring else -10
    zv = sample.StanSamplingVariable(z, "GEV", censor,
                                     ninits=stan_nchains)

    sv = sample.StanSamplingDataset([yv, zv], copula)
    stan_data = sv.to_dict()
    stan_inits = sv.initial_parameters

    fout_stan = FTESTS / "sampling" / "bivariate" / f"{stationid}_{copula}"
    fout_stan.mkdir(exist_ok=True, parents=True)
    for f in fout_stan.glob("*.*"):
        f.unlink()

    smp = bivariate_censored_sampling(data=stan_data,
                                      chains=stan_nchains,
                                      seed=SEED,
                                      iter_warmup=stan_nwarm,
                                      iter_sampling=
                                      stan_nsamples//stan_nchains,
                                      output_dir=fout_stan,
                                      inits=stan_inits,
                                      show_progress=False)
    diag = report.process_stan_diagnostic(smp.diagnose())
    params = smp.draws_pd()

    crs = ["treedepth", "ebfmi", "effsamplesz"]
    for cr in crs:
        assert diag[cr] == "satisfactory"

    # Clean folder
    for f in fout_stan.glob("bivariate_censored*"):
        f.unlink()

    fout_stan.rmdir()


def test_bivariate_sampling(allclose):
    # Same background than univariate sampling tests
    pytest.skip("TODO")

    stationids = get_stationids()
    nstations = 2
    LOGGER = sample.get_logger(stan_logger=False)

    copula_names = sample.COPULA_NAMES
    plots = {i: n for i, n in enumerate(copula_names)}

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 100
    nrepeat = 50
    nrows, ncols = 2, 2
    axwidth, axheight = 5, 5

    for isite in range(nstations):
        # Create stan variables
        y = get_ams(stationids[isite])
        z = get_ams(stationids[isite+1])
        N = len(y)

        z.iloc[-2] = np.nan # to add a missing data in z
        df = pd.DataFrame({"y": y, "z": z}).sort_index()
        y, z = df.y, df.z

        yv = sample.StanSamplingVariable(y, "GEV", 100)
        zv = sample.StanSamplingVariable(z, "GEV", 100)

        # Setup image
        plt.close("all")
        mosaic = [[plots.get(ncols*ir+ic, ".") for ic in range(ncols)]\
                            for ir in range(nrows)]

        w, h = axwidth*ncols, axheight*nrows
        fig = plt.figure(figsize=(w, h), layout="tight")
        axs = fig.subplot_mosaic(mosaic)

        for cop in cops:
            pass


