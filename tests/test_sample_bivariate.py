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
@pytest.mark.parametrize("censoring", [False, True])
def test_bivariate_sampling_satisfactory(copula, censoring, allclose):
    LOGGER = sample.get_logger(stan_logger=True)

    stan_nwarm = 10000
    stan_nsamples = 5000
    stan_nchains = 5

    stationid = STATIONIDS[0]
    y = get_ams(stationid)

    # Generate random covariate
    scale = np.nanstd(y) / 5
    z = y + np.random.normal(0, scale, size=len(y))

    N = len(y)

    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    y, z = df.y, df.z

    censor = y.median() if censoring else np.nanmin(y) - 1.
    yv = sample.StanSamplingVariable(y, "GEV", censor,
                                     ninits=stan_nchains)
    censor = z.median() if censoring else np.nanmin(z) - 1.
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
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    # Test diag
    assert diag["treedepth"] == "satisfactory"
    assert diag["ebfmi"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"

    # Test divergence
    prc = diag["divergence_proportion"]
    if censoring:
        assert prc < 10
    else:
        assert prc < 1

    # Clean folder
    for f in fout_stan.glob("*.*"):
        f.unlink()

    fout_stan.rmdir()
