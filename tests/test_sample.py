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

from nrivfloodfreq import fdist, fsample
from nrivfloodfreqstan import sample, test_marginal, test_copula
from hydrodiy.io import csv

from nrivfloodfreqstan import copulas

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

    for key in ["loc", "logscale", "shape"]:
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

    for stationid in stationids:
        y = get_ams(stationid)
        ymin, ymax = y.min(), y.max()
        N = len(y)

        #for marginal in ["LogNormal", "Gumbel", "GEV", "LogPearson3", "Normal"]:
        for marginal in ["LogPearson3"]:
            dist = fdist.factory(marginal)
            params, _ = fsample.bootstrap_lh_moments(dist, y, 50)
            tbar = tqdm(params.iterrows(), total=len(params), \
                            desc=f"[{stationid}] Testing {marginal} marginal")
            for _, param in tbar:
                dist.set_dict_params(param.to_dict())

                if marginal == "LogNormal":
                    loc = dist.m
                    logscale = math.log(dist.s)
                    shape = 0
                elif marginal == "Normal":
                    loc = dist.mu
                    logscale = dist.logsig
                    shape = 0
                elif marginal == "Gumbel":
                    loc = dist.tau
                    logscale = dist.logalpha
                    shape = 0
                elif marginal == "GEV":
                    loc = dist.tau
                    logscale = dist.logalpha
                    shape = dist.kappa
                elif marginal == "LogPearson3":
                    loc = dist.m
                    logscale = math.log(dist.s)
                    shape = dist.g

                y0, y1 = dist.support
                if y0>ymin or y1<ymax:
                    continue

                stan_data = {
                    "ymarginal": sample.MARGINAL_CODES[marginal], \
                    "N": N, \
                    "y": y.values, \
                    "yloc": loc, \
                    "ylogscale": logscale, \
                    "yshape": shape
                }

                # Run stan
                smp = test_marginal.sample(data=stan_data, \
                                    chains=1, iter_warmup=0, iter_sampling=1, \
                                    fixed_param=True, show_progress=False)
                smp = smp.draws_pd().squeeze()

                # Test
                luncens = smp.filter(regex="luncens").values
                expected = dist.logpdf(y)
                assert allclose(luncens, expected, atol=1e-5)

                lcens = smp.filter(regex="lcens").values
                expected = dist.logcdf(y)
                assert allclose(lcens, expected, atol=1e-5)


def test_copulas():
    N = 100
    uv = np.random.uniform(0, 1, size=(N, 2))
    LOGGER = sample.get_logger(level="INFO") #, stan_logger=False)

    #for copula in ["Gumbel", "Clayton", "Gaussian"]:
    for copula in ["Gaussian"]:
        cop = copulas.factory(copula)

        rmin = cop.rho_min
        rmax = cop.rho_max
        nval = 20
        tbar = tqdm(np.linspace(rmin, rmax, nval), \
                        desc=f"Testing "+copula+" vs statsmodels", total=nval)
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

            # Test lpdf
            lpdf = smp.filter(regex="luncens")
            expected = cop.logpdf(uv)

            # test lcdf
            expected = cop.logcdf(uv)

            # test conditional density
            expected = cop.logconditional_density(uv[:, 0], uv[:, 1])



