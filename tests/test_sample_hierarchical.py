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
from floodstan import hierarchical_censored_nospace_sampling

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

    with pytest.raises(ValueError, match="Expected u_alpha"):
        stan_data = hv.to_dict([0.5] * 4)

    with pytest.raises(ValueError, match="Expected all u_alpha"):
        stan_data = hv.to_dict([1.5] * 3)

    ua = [0.5] * 3
    stan_data = hv.to_dict(ua)

    assert len(stan_data) == 24

    assert not stan_data["shape_has_hierarchical"]

    N = stan_data["N"]
    M = stan_data["M"]

    assert len(stan_data["y"]) == M
    for y in stan_data["y"]:
        assert len(y) == N

    assert len(stan_data["idx_obs"]) == M
    for idx in stan_data["idx_obs"]:
        assert len(idx) == N

    for i in range(M):
        no = stan_data["Nobs"][i]
        ii = np.array(stan_data["idx_obs"][i][:no]) - 1
        yi = np.array(stan_data["y"][i])
        cens = stan_data["ycensors"][i]
        assert (yi >= cens).sum() == no
        assert np.all(yi[ii] >= cens)

    inits = hv.inits()
    assert len(inits) == 10
    for init in inits:
        assert len(init) == 7

    # Check we can save data to json
    ftmp =  FTESTS / "hierarchical_data.json"
    if ftmp.exists():
        ftmp.unlink()

    with ftmp.open("w") as fo:
        json.dump(stan_data, fo, indent=4)
    ftmp.unlink()

    with ftmp.open("w") as fo:
        json.dump(inits, fo, indent=4)
    ftmp.unlink()


@pytest.mark.parametrize("nospace", [True, False])
@pytest.mark.parametrize("shape_has_hierarchical", [False])
@pytest.mark.parametrize("marginal_name",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("censoring", [True])
def test_hierarchical_censored_sampling(nospace, shape_has_hierarchical,
                                        marginal_name, censoring, allclose):
    if marginal_name not in ["GEV", "Gumbel"]:
        pytest.skip(f"Skipping {marginal_name}.")

    y, areas, coords = get_data()
    pcensor = 0.3 if censoring else 0.
    marginal = marginals.factory(marginal_name)

    # Set STAN
    stan_nwarm = 10000
    stan_nsamples = 10000
    stan_nchains = 5

    # Prepare sampling data
    hv = sample.StanHierarchicalDataset(marginal, y, pcensor,
                                        areas, coords,
                                        shape_has_hierarchical,
                                        ninits=stan_nchains)
    stan_data = hv.to_dict([0.9] * 3)
    stan_inits = hv.inits()
    stan_args = hv.stan_sample_args

    # Clean output folder
    fname = f"hierarchical_{marginal_name}_{censoring}"
    fout = FTESTS / "sampling" / fname
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    progress = False
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              parallel_chains=stan_nchains,
              show_progress=progress,
              iter_warmup=stan_nwarm)
    kw.update(stan_args)

    # Sample
    if nospace:
        smp = hierarchical_censored_nospace_sampling(**kw)
    else:
        smp = hierarchical_censored_sampling(**kw)

    df = smp.draws_pd()
    print(f"\n[test hierarchical] {marginal_name} C:{censoring}"
          + f" NS:{nospace} SH:{shape_has_hierarchical}")

    yshape1 = df.filter(regex="^yshape1\\[", axis=1)
    print(f"\tShape std = {yshape1.mean().std():0.4f}")
    if not nospace:
        rho = df.filter(regex="^rho\\[", axis=1)
        rhom = rho.mean().values
        print(f"\tRho mean = {rhom[0]:0.2f} {rhom[1]:0.2f} {rhom[2]:0.2f}")

    tau = np.sqrt(df.filter(regex="^tau2\\[", axis=1))
    taum = tau.mean()
    print(f"\tTau mean = {taum[0]:0.2f} {taum[1]:0.2f} {taum[2]:0.2f}")

    dd = smp.diagnose()
    diag = report.process_stan_diagnostic(dd)
    rhat = smp.summary().loc[:, "R_hat"]

    for f in fout.glob("*.*"):
        f.unlink()
    fout.rmdir()

    # Test diag
    assert diag["effsamplesz"] == "satisfactory"
    assert diag["rhat"] == "satisfactory"
    print(f"\tR_hat = [{rhat.min():0.3f}, {rhat.max():0.3f}]")

    # Test divergence
    prc = diag["divergence_proportion"]
    print(f"\tDivergence proportion = {prc}\n")
    thresh = 50
    assert prc < thresh

    # Test report
    for i in range(stan_data["M"]):
        pp = df.filter(regex=f"\\[{i + 1}\\]$", axis=1)
        rep, _ = report.ams_report(marginal, pp)
        rep = rep.filter(regex="DESIGN", axis=0)
        rep = rep.filter(regex="MEAN|PREDICTIVE", axis=1)
        assert rep.notnull().all().all()


def test_hierarchical_censored_sampling_big(allclose):
    marginal = marginals.factory("GEV")
    pcensor = 0.3

    # Generate random data
    y, areas, coords = get_data()
    y_big = []
    areas_big = []
    coords_big = []
    for repeat in range(4):
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
    stan_nwarm = 5000
    stan_nsamples = 5000
    stan_nchains = 5

    # Prepare sampling data
    hv = sample.StanHierarchicalDataset(marginal, y_big, pcensor,
                                        areas_big, coords_big,
                                        ninits=stan_nchains)
    stan_data = hv.to_dict([0.9] * 3)
    stan_inits = hv.inits()
    stan_args = hv.stan_sample_args

    # Clean output folder
    fname = f"hierarchical_big_{marginal.name}"
    fout = FTESTS / "sampling" / fname
    fout.mkdir(parents=True, exist_ok=True)
    for f in fout.glob("*.*"):
        f.unlink()

    # Sample arguments
    progress = False
    kw = dict(data=stan_data,
              seed=SEED,
              iter_sampling=stan_nsamples // stan_nchains,
              output_dir=fout,
              inits=stan_inits,
              chains=stan_nchains,
              show_progress=progress,
              parallel_chains=stan_nchains,
              iter_warmup=stan_nwarm)
    kw.update(stan_args)

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


