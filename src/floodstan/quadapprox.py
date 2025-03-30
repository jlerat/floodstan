import warnings
import numpy as np

XLARGE = 1e20


def _to_1d(*args):
    if len(args) == 1:
        return np.atleast_1d(args[0])
    else:
        return [np.atleast_1d(v) for v in args]


def _check_xi_a_b_c(xi, a, b, c):
    if np.any(np.diff(xi) <= 0):
        errmess = "Expected xi to be strictly increasing."
        raise ValueError(errmess)

    if len(xi) != len(a) - 1:
        errmess = "Expected len(xi) = len(a) - 1."
        raise ValueError(errmess)

    if len(a) != len(b) or len(a) != len(c):
        errmess = "Expected len of a, b, c to be equal."
        raise ValueError(errmess)


def _expand_xi(xi, min_x, max_x):
    xi = _to_1d(xi)
    xmin = min(xi[0] - 1, min_x)
    xmax = max(xi[-1] + 1, max_x + 1e-10)
    return np.insert(xi, [0, len(xi)], [xmin, xmax])


def get_coefficients(xi, fi, fm, monotonous=False):
    xi, fi, fm = _to_1d(xi, fi, fm)
    if len(xi) != len(fi):
        errmess = "Expected len(xi) = len(fi)."
        raise ValueError(errmess)

    if len(xi) - 1 != len(fm):
        errmess = "Expected len(xi) = len(fm) + 1."
        raise ValueError(errmess)

    if np.any(np.diff(xi) <= 0):
        errmess = "Expected xi to be strictly increasing."
        raise ValueError(errmess)

    x0, f0 = xi[:-1], fi[:-1]
    x1, f1 = xi[1:], fi[1:]

    # only monotonous functions
    if monotonous:
        v0 = (3 * f0 + f1) / 4
        v1 = (f0 + 3 * f1) / 4
        fm_low = np.minimum(v0, v1)
        fm_high = np.maximum(v0, v1)
        fm = np.clip(fm, fm_low, fm_high)

    # Coefs f = a * x**2 + b x + c
    dx = x1 - x0
    a = (2. * f0 + 2. * f1 - 4. * fm) / dx**2
    phi = (4. * fm - 3. * f0 - f1) / dx
    b = -2. * x0 * a + phi
    c = x0**2 * a - x0 * phi + f0

    # Prolonge fitting linearly above min and below max node
    df0 = 2 * a[0] * x0[0] + b[0]
    df1 = 2 * a[-1] * x1[-1] + b[-1]
    n = len(a)
    a = np.insert(a, [0, n], [0, 0])
    b = np.insert(b, [0, n], [df0, df1])
    c = np.insert(c, [0, n], [f0[0] - x0[0] * df0,
                              f1[-1] - x1[-1] * df1])
    return a, b, c


def forward(x, xi, a, b, c):
    x, xi, a, b, c = _to_1d(x, xi, a, b, c)
    _check_xi_a_b_c(xi, a, b, c)

    min_x = np.nanmin(x)
    max_x = np.nanmax(x)
    xi = _expand_xi(xi, min_x, max_x)

    x0, x1 = xi[:-1], xi[1:]
    xv = x[:, None]
    u = (xv - x0) / (x1 - x0)
    isin = (u >= 0) & (u < 1)
    fhat = a[None, :] * xv * xv + b[None, :] * xv + c[None, :]
    return np.einsum("ij,ij->i", fhat, isin)


def inverse(f, xi, a, b, c):
    f, xi, a, b, c = _to_1d(f, xi, a, b, c)
    _check_xi_a_b_c(xi, a, b, c)
    xi = _expand_xi(xi, -XLARGE, XLARGE)

    # Quadratic equation roots
    c_root = c - f[:, None]
    delta = b**2 - 4. * a * c_root
    ineg = delta < 0
    delta[ineg] = 0.
    q = -(b + np.sign(b) * np.sqrt(delta)) / 2.
    q[ineg] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xr0 = np.nan_to_num(q / a, nan=np.inf)
        xr1 = np.nan_to_num(c_root / q, nan=np.inf)

    # Eliminate linear extrapolation
    xr0[:, [0, -1]] = np.inf
    xr1[:, [0, -1]] = np.inf

    # Check roots are within interpolation bands
    x0, x1 = xi[:-1], xi[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ur0 = (xr0 - x0) / (x1 - x0)

    isin = np.where((ur0 >= 0) & (ur0 <= 1))
    idx, xtmp = [], []
    if len(isin[0]) > 0:
        idx.append(isin[0])
        xtmp.append(xr0[isin])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ur1 = (xr1 - x0) / (x1 - x0)

    isin = np.where((ur1 >= 0) & (ur1 <= 1))
    if len(isin[0]) > 0:
        idx.append(isin[0])
        xtmp.append(xr1[isin])

    idx = np.concatenate(idx)
    xtmp = np.concatenate(xtmp)

    xr = np.full(f.shape, fill_value=np.nan)
    xr[idx] = xtmp
    return xr
