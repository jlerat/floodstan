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
from floodstan import bivariate_censored_sampling

from test_sample_univariate import get_stationids, get_ams, TQDM_DISABLE

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_bivariate_sampling(allclose):
    # Same background than univariate sampling tests
    return
    # TODO !

    stationids = get_stationids()
    nstations = 2
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

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


