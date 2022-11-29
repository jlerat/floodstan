import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import zipfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, kstest
from scipy.stats import lognorm
from scipy.linalg import toeplitz

import pytest
import warnings

from cmdstanpy import CmdStanModel

import importlib
from tqdm import tqdm

from nrivfloodfreq import fdist, fsample
from nrivfloodfreqstan import sample, test_stan_functions
from hydrodiy.io import csv

#import data_reader

np.random.seed(5446)

FTESTS = Path(__file__).resolve().parent

# --- Utils functions ----------------------------
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
    prior = sample.get_marginal_priors("streamflow_obs", "GEV", \
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
    assert pd.notnull(df.y[i11]).all()
    assert pd.notnull(df.z[i11]).all()

    i31 = stan_data["i31"]
    assert pd.isnull(df.y[i31]).all()
    assert pd.notnull(df.z[i31]).all()


def test_marginals():
    y = get_ams("203014")
    N = len(y)
    prior_variables = {"area": 500}

    for marginal in ["LogNormal", "Gumbel", "GEV", "LogPearson3"]:
        dist = fdist.factory(marginal)
        params = dist.fit_lh_moments(y).loc[0]
        dist.set_dict_params(params.to_dict())

        for i in range(N):
            stan_data = sample.prepare(y.values[[i]], \
                                    ymarginal=marginal, \
                                    zmarginal="Normal", \
                                    prior_variables=prior_variables)
            stan_data["zloc"] = 10
            stan_data["zlogshape"] = 0
            stan_data["zshape"] = 0
            stan_data["rho"] = 0.5

            if marginal == "LogNormal":
                loc = dist.m
                logscale = math.log(dist.s)
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
                loc = dist.s
                logscale = math.log(dist.s)
                shape = dist.g

            stan_data["yloc"] = loc
            stan_data["ylogscale"] = logscale
            stan_data["yshape"] = shape

            smp = test_stan_functions.sample(data=stan_data, \
                                chains=1, iter_warmup=0, iter_sampling=1)
            import pdb; pdb.set_trace()


