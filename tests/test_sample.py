import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal

import pytest
import warnings

from cmdstanpy import CmdStanModel

import importlib
from tqdm import tqdm

from nrivfloodfreqstan import marginals, sample, copulas
from nrivfloodfreqstan import test_marginal, test_copula

from tqdm import tqdm

np.random.seed(5446)

FTESTS = Path(__file__).resolve().parent

# --- Utils functions ----------------------------
def get_stationids():
    fs = FTESTS / "data"
    stationids = []
    for f in fs.glob("*_AMS.csv"):
        stationids.append(re.sub("_.*", "", f.stem))
    return stationids

def get_ams(stationid):
    fs = FTESTS / "data" / f"{stationid}_AMS.csv"
    df = pd.read_csv(fs, skiprows=15, index_col=0)
    return df.iloc[:, 0]


# ------------------------------------------------

def test_get_stan_model():
    for modname in sample.STAN_MODEL_NAMES:
        model = sample.load_stan_model(modname)
        print(f"\n{modname}: {model.exe_info()}\n")


def test_get_copula_prior():
    prior = sample.get_copula_prior("uninformative")
    assert np.allclose(prior, [0.8, 1])


def test_get_marginal_prior():
    prior_variables = {"area": 100}
    prior = sample.get_marginal_prior("streamflow_obs", "GEV", \
                                prior_variables, "uninformative")

    expected = {\
        "loc": [700./3, 1400./3], \
        "logscale": [5, 4], \
        "shape": [0, 4]
    }

    for key in ["locn", "logscale", "shape"]:
        assert np.allclose(prior[key], expected[key])



def test_prepare():
    y = get_ams("203010")
    z = get_ams("203014")
    df = pd.DataFrame({"y": y, "z": z})
    prior_variables = {"area": 500}
    stan_data = sample.prepare(df.y, df.z, prior_variables=prior_variables)

    Ncases = stan_data["Ncases"]
    assert np.allclose(Ncases, [[55, 0], [0, 0], [10, 0]])

    i11 = stan_data["i11"]
    assert pd.notnull(df.y.iloc[i11-1]).all()
    assert pd.notnull(df.z.iloc[i11-1]).all()

    i31 = stan_data["i31"]
    assert pd.isnull(df.y.iloc[i31-1]).all()
    assert pd.notnull(df.z.iloc[i31-1]).all()


def test_marginals(allclose):
    stationids = get_stationids()
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)
    nboot = 100

    for stationid in stationids:
        y = get_ams(stationid)
        N = len(y)

        for marginal in ["LogNormal", "Gumbel", "GEV", "LogPearson3", "Normal"]:
            dist = marginals.factory(marginal)
            desc = f"[{stationid}] Testing stan {marginal} marginal"

            for iboot in tqdm(range(nboot), desc=desc):
                # Bootstrap fit
                yboot = np.random.choice(y.values, N)
                dist.fit_lh_moments(yboot)
                y0, y1 = dist.support
                if y0>yboot.min() or y1<yboot.max():
                    continue

                stan_data = {
                    "ymarginal": sample.MARGINAL_CODES[marginal], \
                    "N": N, \
                    "y": yboot, \
                    "ylocn": dist.locn, \
                    "ylogscale": dist.logscale, \
                    "yshape1": dist.shape1
                }

                # Run stan
                smp = test_marginal.sample(data=stan_data, \
                                    chains=1, iter_warmup=0, iter_sampling=1, \
                                    fixed_param=True, show_progress=False)
                smp = smp.draws_pd().squeeze()

                # Test
                luncens = smp.filter(regex="luncens").values
                expected = dist.logpdf(yboot)
                assert allclose(luncens, expected, atol=1e-5)

                lcens = smp.filter(regex="lcens").values
                expected = dist.logcdf(yboot)
                assert allclose(lcens, expected, atol=1e-5)


def test_copulas(allclose):
    N = 100
    uv = np.random.uniform(0, 1, size=(N, 2))
    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    #for copula in ["Gumbel", "Clayton", "Gaussian"]:
    for copula in ["Gaussian"]:
        cop = copulas.factory(copula)

        rmin = cop.rho_min
        rmax = cop.rho_max
        nval = 20
        tbar = tqdm(np.linspace(rmin, rmax, nval), \
                        desc=f"Testing stan "+copula, total=nval)
        for rho in tbar:
            cop.rho = rho

            stan_data = {
                "copula": sample.COPULA_CODES[copula], \
                "N": N, \
                "uv": uv, \
                "rho": rho
            }

            # Run stan
            smp = test_copula.sample(data=stan_data, \
                                chains=1, iter_warmup=0, iter_sampling=1, \
                                fixed_param=True, show_progress=False)
            smp = smp.draws_pd().squeeze()

            assert allclose(smp.rho_check, rho, atol=1e-6)

            # Test lpdf
            lpdf = smp.filter(regex="luncens")
            expected = cop.logpdf(uv)
            assert allclose(lpdf, expected, atol=1e-7)

            # test lcdf
            lcdf = smp.filter(regex="lcens")
            expected = cop.logcdf(uv)
            assert allclose(lcdf, expected, atol=1e-7)
            lcond = smp.filter(regex="lcond")
            expected = np.log(cop.conditional_density(uv[:, 0], uv[:, 1]))
            assert allclose(lcond, expected, atol=1e-7)



