import sys, re, math, json
from io import StringIO
import numpy as np
import pandas as pd
from itertools import product as prod
from scipy.stats import norm, poisson, bernoulli, nbinom
from pathlib import Path
import pytest
import warnings

from floodstan import discretes

from tqdm import tqdm

SEED = 5446

FTESTS = Path(__file__).resolve().parent


def test_discretedist(allclose):
    name = "bidule"
    dist = discretes.DiscreteDistribution(name)

    assert dist.name == name
    assert hasattr(dist, "locn")
    assert hasattr(dist, "phi")

    s = str(dist)
    assert isinstance(s, str)

    dist.locn = 10
    assert dist.locn == 10

    msg = "Expected locn in"
    with pytest.raises(AssertionError, match=msg):
        dist.locn = -1

    msg = "Expected phi in"
    with pytest.raises(AssertionError, match=msg):
        dist.phi = -1

    for pn in discretes.PARAMETERS:
        dist[pn] = 1
        assert getattr(dist, pn) == 1

        dist[pn] = 2
        assert dist[pn] == 2


def test_discretes_vs_scipy(allclose):
    ntests = 10
    N = 100

    for dname, dcode in discretes.DISCRETE_NAMES.items():
        locn_mu = 0.5 if dname == "Bernoulli" else 3
        locn_max = 1 if dname == "Bernoulli" else 20
        locn_sig = 0.1 if dname == "Bernoulli" else 1
        phi_mu, phi_sig = 1, 1

        for i in range(ntests):
            p0, p1 = norm.cdf([0, locn_max], loc=locn_mu, scale=locn_sig)
            u = np.random.uniform()
            p = p0+(p1-p0)*u
            klocn = norm.ppf(p)*locn_sig+locn_mu

            p0, p1 = norm.cdf([1e-3, 3], loc=phi_mu, scale=phi_sig)
            u = np.random.uniform()
            p = p0+(p1-p0)*u
            kphi = norm.ppf(p, loc=phi_mu, scale=phi_sig)

            if dname == "Poisson":
                rcv = poisson(mu=klocn)
            elif dname == "Bernoulli":
                rcv = bernoulli(p=klocn)
            else:
                # reparameterize as per
                # https://mc-stan.org/docs/functions-reference/nbalt.html
                v = klocn+klocn**2/kphi
                n = klocn**2/(v-klocn)
                p = klocn/v
                rcv = nbinom(n=n, p=p)

            kv = discretes.factory(dname)
            kv.locn = klocn
            kv.phi = kphi
            x = np.arange(10)

            for fun in ["pmf", "logpmf", "cdf", "logcdf"]:
                f1 = getattr(kv, fun)
                f2 = getattr(rcv, fun)
                assert allclose(f1(x), f2(x))

            if kv.isbern:
                continue

            # test pot 2 cdf ams
            pot = 0.99
            ams = kv.pot2ams_cdf(pot)


