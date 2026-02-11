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
from floodstan import sample
from floodstan import copulas
from floodstan import report
from floodstan import load_stan_model
from floodstan import hierarchical_censored_sampling

from utils import get_stationids, get_ams, get_info
from utils import FTESTS

SEED = 5446
np.random.seed(SEED)

LOGGER = sample.get_logger(stan_logger=False)

def get_data():
    sids = ["203010", "203014", "203024", "203004", "203002"]
    y = pd.DataFrame({sid: get_ams(sid) for sid in sids})
    y = y.values

    areas = np.array([179., 223., 148., 1790.,  62.])

    coords = np.array([
        [153.16, -28.74],
        [153.39, -28.76],
        [153.36, -28.72],
        [153.06, -28.86],
        [153.41, -28.64],
        ])
    coords *= 4e4 / 365
    return y, areas, coords


def test_stan_sampling_variable(allclose):
    y, areas, coords = get_data()
    pcensor = 0.3
    marginal = marginals.factory("GEV")
    hv = sample.StanHierarchicalDataset(marginal, y, pcensor,
                                        areas, coords)

    stan_data = hv.to_dict()
    assert len(stan_data) == 25
    N = stan_data["N"]
    M = stan_data["M"]
    assert stan_data["y"].shape == (M, N)
    assert stan_data["idx_obs"].shape == (M, N)

    for i in range(M):
        no = stan_data["Nobs"][i]
        ii = stan_data["idx_obs"][i][:no] - 1
        yi = stan_data["y"][i]
        cens = stan_data["ycensors"][i]
        assert (yi >= cens).sum() == no
        assert np.all(yi[ii] >= cens)

    inits = hv.inits()
    assert len(inits) == 8


@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("censoring", [False, True])
def test_hierarchical_censored_sampling(marginal_name, censoring, allclose):
    if marginal_name in ["Gamma", "LogPearson3", "GeneralizedPareto"]:
        pytest.skip(f"Skipping {marginal_name}.")

    y, areas, coords = get_data()
    pcensor = 0.3 if censoring else 0.
    marginal = marginals.factory(marginal_name)

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 10000
    if marginal_name == "GeneralizedPareto":
        stan_nchains = 10
    else:
        stan_nchains = 5

    # Prepare sampling data
    hv = sample.StanHierarchicalDataset(marginal, y, pcensor,
                                        areas, coords)
    stan_data = hv.to_dict()
    stan_inits = hv.inits()

    # Clean output folder
    fname = f"hierarchical_{marginal_name}_{censoring}"
    fout = FTESTS / "sampling" / fname
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm)

    # Sample
    smp = hierarchical_censored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    for f in fout.glob("*.*"):
        f.unlink()
    fout.rmdir()

    # Test diag
    assert diag["effsamplesz"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"

    # Test divergence
    prc = diag["divergence_proportion"]
    print(f"\n{stationid}-{marginal_name}-{censoring} : Divergence proportion = {prc}\n")
    if not marginal.name in ["LogPearson3", "GeneralizedPareto",
                             "GeneralizedLogistic"]:
        thresh = 50
        assert prc < thresh

    # Test report
    rep, _ = report.ams_report(marginal, df)
    rep = rep.filter(regex="DESIGN", axis=0)
    rep = rep.filter(regex="MEAN|PREDICTIVE", axis=1)
    assert rep.notnull().all().all()


def test_hierarchical_censored_sampling_big(allclose):
    marginal = marginals.factory("GEV")
    pcensor = 0.3

    y, areas, coords = get_data()
    y_big = []
    areas_big = []
    coords_big = []
    for repeat in range(40):
        err = np.random.uniform(0.8, 1.2, size=y.shape)
        y_big.append(y * err)

        err = np.random.uniform(0.8, 1.2, size=areas.shape)
        areas_big.append(areas * err)

        err = np.random.uniform(0.8, 1.2, size=coords.shape)
        coords_big.append(coords* err)

    y_big = np.column_stack(y_big)
    areas_big = np.concatenate(areas_big)
    coords_big = np.row_stack(coords_big)

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 10000
    stan_nchains = 5

    # Prepare sampling data
    hv = sample.StanHierarchicalDataset(marginal, y_big, pcensor,
                                        areas_big, coords_big)
    stan_data = hv.to_dict()
    stan_inits = hv.inits()

    # Clean output folder
    fname = f"hierarchical_big_{marginal.name}"
    fout = FTESTS / "sampling" / fname
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm)

    # Sample
    smp = hierarchical_censored_sampling(**kw)
    df = smp.draws_pd()
    diag = report.process_stan_diagnostic(smp.diagnose())

    for f in fout.glob("*.*"):
        f.unlink()
    fout.rmdir()

    # Test diag
    assert diag["effsamplesz"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"


