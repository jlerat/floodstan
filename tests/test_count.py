import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import poisson, nbinom

import pytest
import warnings

from hydrodiy.plot import putils

import importlib
from tqdm import tqdm

from floodstan.sample import COUNT_NAMES
from floodstan import test_count_modelling, count_modelling

from test_sample import get_stationids, get_ams

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_count_vs_stan(allclose):
    ntests = 10
    N = 100

    for i in range(ntests):
        ylocn = np.random.normal(loc=3, scale=1)
        yphi = np.random.normal(loc=1, scale=0.5)
        if ylocn<1e-3 or yphi<1e-3:
            continue

        for cname, ccode in COUNT_NAMES.items():
            if cname == "Poisson":
                rcv = poisson(mu=ylocn)
            else:
                # reparameterize as per
                # https://mc-stan.org/docs/functions-reference/nbalt.html
                v = ylocn+ylocn**2/yphi
                n = ylocn**2/(v-ylocn)
                p = ylocn/v
                rcv = nbinom(n=n, p=p)

            yn = rcv.rvs(size=N)
            stan_data = {
                "N": N, \
                "ycount": ccode, \
                "yn": yn, \
                "ylocn": ylocn, \
                "yphi": yphi
            }

            # Run stan
            smp = test_count_modelling.sample(data=stan_data, \
                                chains=1, iter_warmup=0, iter_sampling=1, \
                                fixed_param=True, show_progress=False)
            smp = smp.draws_pd().squeeze()

            # Test
            lpmf = smp.filter(regex="lpmf").values
            expected = rcv.logpmf(yn)
            assert allclose(lpmf, expected, atol=1e-5)


