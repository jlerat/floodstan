import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import poisson, nbinom, bernoulli

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
    e = poisson(mu=2.5).rvs(size=100)
    sv = sample.StanDiscreteVariable()

    msg = "Data is not set."
    with pytest.raises(ValueError, match=msg):
        d = sv.data

    msg = "Cannot find"
    with pytest.raises(AssertionError, match=msg):
        sv.set(e, "bidule")

    msg = "Expected data"
    with pytest.raises(AssertionError, match=msg):
        sv.set(e[:, None], "Poisson")

    en = e.copy()
    en[:3] = -1
    msg = "Need all data"
    with pytest.raises(AssertionError, match=msg):
        sv.set(en, "Poisson")

    en = e.copy()
    en[:3] = 11
    msg = "Need all data"
    with pytest.raises(AssertionError, match=msg):
        sv.set(en, "Poisson")

    for dname in sample.DISCRETE_NAMES:
        en = e.clip(0, 1) if dname == "Bernoulli" else e
        sv.set(en, dname)
        assert allclose(sv.data, en)
        assert sv.N == len(e)
        assert sv.discrete_code == sample.DISCRETE_NAMES[dname]
        assert sv.discrete_name == dname

        dd = sv.to_dict()
        assert "edisc" in dd
        assert "e" in dd

    # Rapid setting
    sv = sample.StanDiscreteVariable(e)
    msg = "Data is not set."
    with pytest.raises(ValueError, match=msg):
        d = sv.data


def test_discrete_vs_stan(allclose):
    ntests = 10
    N = 100

    for i in range(ntests):
        for dname, dcode in sample.DISCRETE_NAMES.items():
            loc = 0.5 if dname == "Bernoulli" else 3
            scale = 0.1 if dname == "Bernoulli" else 1
            elocn = np.random.normal(loc=loc, scale=scale)
            ephi = np.random.normal(loc=1, scale=0.5)
            if elocn<1e-3 or ephi<1e-2:
                continue

            if dname == "Poisson":
                rcv = poisson(mu=elocn)
            elif dname == "Bernoulli":
                rcv = bernoulli(p=elocn)
            else:
                # reparameterize as per
                # https://mc-stan.org/docs/functions-reference/nbalt.html
                v = elocn+elocn**2/ephi
                n = elocn**2/(v-elocn)
                p = elocn/v
                rcv = nbinom(n=n, p=p)

            e = rcv.rvs(size=N).clip(0, sample.NEVENT_UPPER)
            ev = sample.StanDiscreteVariable(e, dname)
            stan_data = ev.to_dict()
            stan_data["elocn"] = elocn
            stan_data["ephi"] = ephi

            # Run stan
            smp = test_discrete.sample(data=stan_data, \
                                chains=1, iter_warmup=0, iter_sampling=1, \
                                fixed_param=True, show_progress=False)
            smp = smp.draws_pd().squeeze()

            # Test
            lpmf = smp.filter(regex="lpmf").values
            expected = rcv.logpmf(e)
            assert allclose(lpmf, expected, atol=1e-5)


