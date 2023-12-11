import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, poisson, nbinom, bernoulli

import pytest
import warnings

import importlib
from tqdm import tqdm

from floodstan import sample
from floodstan import test_discrete, event_occurrence

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_stan_discrete_variable(allclose):
    k = poisson(mu=2.5).rvs(size=100)
    msg = "Cannot find"
    with pytest.raises(AssertionError, match=msg):
        sv = sample.StanDiscreteVariable(k, "bidule")

    msg = "Expected data"
    with pytest.raises(AssertionError, match=msg):
        sv = sample.StanDiscreteVariable(k[:, None], "Poisson")

    kn = k.copy()
    kn[:3] = -1
    msg = "Need all data"
    with pytest.raises(AssertionError, match=msg):
        sv = sample.StanDiscreteVariable(kn, "Poisson")

    kn = k.copy()
    kn[:3] = 101
    msg = "Need all data"
    with pytest.raises(AssertionError, match=msg):
        sv = sample.StanDiscreteVariable(kn, "Poisson")


    for dname in sample.DISCRETE_NAMES:
        kn = k.clip(0, 1) if dname == "Bernoulli" else k
        sv = sample.StanDiscreteVariable(kn, dname)
        assert allclose(sv.data, kn)
        assert sv.N == len(k)
        assert sv.discrete_code == sample.DISCRETE_NAMES[dname]
        assert sv.discrete_name == dname

        dd = sv.to_dict()
        assert "kdisc" in dd
        assert "k" in dd



