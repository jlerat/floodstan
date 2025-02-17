import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn, uniform
from scipy.stats import ks_1samp
from scipy.stats import percentileofscore

import warnings

from hydrodiy.plot import putils

from floodstan import marginals
from floodstan import copulas

from floodstan import report, sample
from floodstan import bivariate_censored_sampling

from test_sample_univariate import get_stationids
from test_sample_univariate import get_ams

SEED = 5446

FTESTS = Path(__file__).resolve().parent

STATIONIDS = get_stationids()


def bivariate_sampling_cook(marginal, copula, stationid1, stationid2):
    # Same background than univariate sampling tests

    LOGGER = sample.get_logger(stan_logger=False)

    # Stan config
    stan_nchains = 3 if debug else 10
    stan_nsamples = 1000 if debug else 10000
    stan_warmup = 1000 if debug else 10000

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 50
    nrepeat = 10 if debug else 100

    # Create stan variables
    y = get_ams(stationid1)
    z = get_ams(stationid2)
    N = len(y)

    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    yv = sample.StanSamplingVariable(y, marginal, 100)
    zv = sample.StanSamplingVariable(z, marginal, 100)

    # TODO


def main(marginal, copula, debug):
    stationids = get_stationids()[:3]

    for stationid1, stationid2 in zip(stationids, stationids[1:]):
        bivariate_sampling_cook(marginal, copula, stationid1, stationid2, debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run cook tests for bivariate sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-m", "--marginal", help="Marginal name",
                        type=str, required=True,
                        choices=list(marginals.MARGINAL_NAMES.keys()))
    parser.add_argument("-c", "--copula", help="Copula name",
                        type=str, required=True,
                        choices=list(copulas.COPULA_NAMES.keys()))
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()
    marginal = args.marginal
    debug = args.debug

    main(marginal, copula, debug)
