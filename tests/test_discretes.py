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

def sample_from_norm_truncated(mu, sig, x0, x1, size=None):
    assert x0<x1
    p0, p1 = norm.cdf([x0, x1], loc=mu, scale=sig)
    u = np.random.uniform(size=size)
    p = p0+(p1-p0)*u
    return mu+sig*norm.ppf(p)


def test_discretedist(allclose):
    name = "bidule"
    dist = discretes.DiscreteDistribution(name)

    assert dist.name == name

    msg = "locn is not set"
    with pytest.raises(ValueError, match=msg):
        dist.locn

    msg = "phi is not set"
    with pytest.raises(ValueError, match=msg):
        dist.phi

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

    s = str(dist)
    assert isinstance(s, str)


def test_discretes_vs_scipy(allclose):
    ntests = 10
    N = 100

    for dname, dcode in discretes.DISCRETE_NAMES.items():
        locn_mu = 0.5 if dname == "Bernoulli" else 3
        locn_max = 1 if dname == "Bernoulli" else 20
        locn_sig = 0.1 if dname == "Bernoulli" else 1
        phi_mu, phi_sig = 1, 1

        for i in range(ntests):
            klocn = sample_from_norm_truncated(locn_mu, locn_sig, 0, locn_max)
            kphi = sample_from_norm_truncated(phi_mu, phi_sig, 1e-1, 3)

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
            for pot in np.linspace(0.5, 0.999, 10):
                ams = kv.pot2ams_cdf(pot)
                expected = kv.ams2pot_cdf(ams)
                assert allclose(pot, expected)

                expected = kv.pot2ams_cdf(expected)
                assert allclose(ams, expected)


            pot = np.linspace(0.5, 0.999, 100)
            ams = kv.pot2ams_cdf(pot)
            expected = kv.ams2pot_cdf(ams)
            assert allclose(pot, expected)

            expected = kv.pot2ams_cdf(expected)
            assert allclose(ams, expected)



def test_discretes_pot2cdf(allclose):
    kv = discretes.factory("Poisson")
    kv.locn = 0.62
    assert np.isnan(kv.ams2pot_cdf(0.51))

    kv = discretes.factory("NegativeBinomial")
    kv.locn = 0.6
    kv.phi = 1
    ams = np.linspace(0.6, 0.7, 10)
    pot = kv.ams2pot_cdf(ams)
    assert np.all(np.isnan(pot[:3]))


