import math
from pathlib import Path

import numpy as np

import pytest
import warnings

from hydrodiy.plot import putils

import importlib

from floodstan import marginals
from floodstan import quadapprox
from floodstan.report import DESIGN_ARIS
from floodstan.report import _prepare_design_aris

from test_sample_univariate import get_stationids, get_ams

SEED = 5446


@pytest.mark.parametrize("inside", [True, False])
def test_forward(inside, allclose):
    if inside:
        x = np.linspace(0.1, 0.9, 100)
    else:
        x = np.linspace(-1, 2, 100)

    xi = [0, 0.5, 1]
    xxi = [-1e10] + xi + [1e10]
    nxi = len(xi) + 1

    for boot in range(10):
        a = np.random.uniform(-2, 2, nxi)
        b = np.random.uniform(-2, 2, nxi)
        c = np.random.uniform(-2, 2, nxi)
        fhat = quadapprox.forward(x, xi, a, b, c)

        for j in range(len(xxi) - 1):
            x0 = xxi[j]
            x1 = xxi[j + 1]
            aj, bj, cj = a[j], b[j], c[j]

            idx = (x >= x0) & (x < x1)
            expected = aj * x**2 + bj * x + cj
            assert allclose(fhat[idx], expected[idx])


@pytest.mark.parametrize("inside", [True, False])
def test_get_coefficients(inside, allclose):
    def fun(x):
        return np.exp(-x**2 / 2) * np.sin(5 * math.pi * x)

    xi = np.linspace(-1, 1, 100)
    fi = fun(xi)
    xm = (xi[:-1] + xi[1:])/2
    fm = fun(xm)

    if inside:
        x = np.linspace(xi[0], xi[-1], 10000)
    else:
        x = np.linspace(-2, 2, 10000)
    f = fun(x)

    a, b, c = quadapprox.get_coefficients(xi, fi, fm)
    fhat = quadapprox.forward(x, xi, a, b, c)

    if inside:
        assert allclose(fhat, f, atol=5e-4, rtol=0)
    else:
        idx = (x >= xi[0]) & (x <= xi[-1])
        assert allclose(fhat[idx], f[idx], atol=5e-4, rtol=0)

        df = 2 * a[[1, -2]] * xi[[0, -1]] + b[[1, -2]]
        v0 = df[0] * (x - xi[0]) + fi[0]
        idx = x < xi[0]
        assert allclose(fhat[idx], v0[idx], atol=5e-4, rtol=0)

        v1 = df[1] * (x - xi[-1]) + fi[-1]
        idx = x > xi[-1]
        assert allclose(fhat[idx], v1[idx], atol=5e-4, rtol=0)


def test_get_coefficients_monotonous(allclose):
    fun = lambda x: x * (1. - x)
    xi = [0, 1]
    fi = [fun(x) for x in xi]
    fm = fun(0.5)
    a, b, c = quadapprox.get_coefficients(xi, fi, fm)


@pytest.mark.parametrize("namefun", ["expsin", "x3", "sin",
                                     "tanh", "exp"])
def test_inverse(namefun, allclose):
    if namefun == "expsin":
        fun = lambda x: np.exp(-x**2 / 2) * np.sin(5 * math.pi * x)
        invfun = None
    elif namefun == "x3":
        fun = lambda x: x**3
        invfun = lambda x: x**(1./3)
    elif namefun == "sin":
        fun = lambda x: np.sin(10 * math.pi * x)
        invfun = None
    elif namefun == "tanh":
        fun = lambda x: np.tanh(x)
        invfun = lambda x: np.arctanh(x)
    elif namefun == "exp":
        fun = lambda x: np.exp(x)
        invfun = lambda x: np.log(x)

    xmin, xmax = -2, 2
    xi = np.linspace(xmin, xmax, 200)
    fi = fun(xi)
    xm = (xi[:-1] + xi[1:])/2
    fm = fun(xm)
    a, b, c = quadapprox.get_coefficients(xi, fi, fm)

    eps = 1e-10
    f = np.linspace(fi.min() + eps, fi.max() - eps, 100)
    x = quadapprox.inverse(f, xi, a, b, c)
    f2 = fun(x)
    assert allclose(f, f2, atol=3e-3)

    if invfun is not None:
        expected = invfun(f)
        iok = ~np.isnan(expected)
        err = np.abs(x[iok] - expected[iok])
        assert allclose(x[iok], expected[iok], atol=1e-4, rtol=0.)



@pytest.mark.parametrize("distname",
                         marginals.MARGINAL_NAMES)
@pytest.mark.parametrize("stationid",
                         get_stationids())
def test_approx_cdf(distname, stationid, allclose):
    streamflow = get_ams(stationid)
    marginal = marginals.factory(distname)
    marginal.params_guess(streamflow)

    design_aris, design_cdf, _, post_pred_cdf = \
        _prepare_design_aris(DESIGN_ARIS, 0.)

    xi = np.unique(marginal.ppf(post_pred_cdf))
    xm = (xi[1:] + xi[:-1]) / 2
    fi = marginal.cdf(xi)
    fm = marginal.cdf(xm)

    a, b, c = quadapprox.get_coefficients(xi, fi, fm)
    q = quadapprox.inverse(design_cdf, xi, a, b, c)
    expected = marginal.ppf(design_cdf)
    emax = np.abs(expected - q).max()
    assert allclose(q, expected, atol=5e-3, rtol=0)
