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
from nrivfloodfreqstan import sfsample
from hydrodiy.io import csv

import data_reader

np.random.seed(5446)

FTESTS = Path(__file__).resolve().parent


def test_get_stan_model():
    #modnames = fsample.STAN_MODEL_NAMES
    modnames = ["bivariate_censoring"]

    for modname in modnames:
        dist = fdist.factory(distname)
        model = sfsample.get_stan_model(modname, True)


