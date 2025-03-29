import math
from pathlib import Path

import numpy as np

import pytest
import warnings

from hydrodiy.plot import putils

import importlib

from floodstan import quadapprox

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

    a, b, c = quadapprox.get_coefficients(xi, fi, fm, False)
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


@pytest.mark.parametrize("namefun", ["expsin", "x3", "sinx"])
def test_inverse(namefun, allclose):
    def fun(x):
        if namefun == "expsin":
            return np.exp(-x**2 / 2) * np.sin(5 * math.pi * x)
        elif namefun == "x3":
            return x**3
        elif namefun == "sinx":
            return np.sin(x)

    xi = np.linspace(-1, 1, 100)
    fi = fun(xi)
    xm = (xi[:-1] + xi[1:])/2
    fm = fun(xm)
    a, b, c = quadapprox.get_coefficients(xi, fi, fm)

    f = np.linspace(fi.min() + 1e-10, fi.max() - 1e-10, 100)
    x = quadapprox.inverse(f, xi, a, b, c)
    f2 = fun(x)
    assert allclose(f, f2, atol=1e-3)



