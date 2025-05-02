import warnings
import numpy as np
from floodstan.data_processing import to1d

XLARGE = 1e20


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
    xi = to1d(xi)
    xmin = min(xi[0] - 1, min_x)
    xmax = max(xi[-1] + 1, max_x + 1e-10)
    return np.insert(xi, [0, len(xi)], [xmin, xmax])


def get_coefficients(xi, fi, fm, monotonous=False):
    xi, fi, fm = [to1d(v) for v in [xi, fi, fm]]
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
    x, xi, a, b, c = [to1d(v) for v in [x, xi, a, b, c]]
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
    f, xi, a, b, c = [to1d(v) for v in [f, xi, a, b, c]]
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

    # Check roots are within interpolation bands
    x0, x1 = xi[:-1], xi[1:]
    xtmp = np.nan * np.zeros_like(xr0)
    cond = np.zeros_like(xr0).astype(bool)
    for xr in [xr0, xr1]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ur = (xr - x0) / (x1 - x0)

        # Checking if root is in interpolation band.
        # If not, it's not a valid root.
        c = (ur >= 0) & (ur <= 1)
        cond[c] = True

        # here we overwrite previous roots to
        # keep the second one only. Could be improved
        xtmp[c] = xr[c]

    # Eliminate extrapolation if possible
    multiple = cond.sum(axis=1) > 1
    noextrapol = np.any(cond[:, 1:-1], axis=1)
    if np.any(noextrapol):
        remove_is_safe = multiple & noextrapol
        if remove_is_safe.sum() > 0:
            # we exclude extrapolated roots when
            # there is an non-extrapolated alternative
            cond[remove_is_safe, 0] = False
            cond[remove_is_safe, -1] = False

    isin = np.where(cond)
    ival = isin[0]
    xr = np.full(f.shape, fill_value=np.nan)

    if len(ival) == 0:
        # No roots found
        return xr

    xr[ival] = xtmp[isin]
    return xr
