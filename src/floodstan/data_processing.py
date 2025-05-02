import numpy as np
import pandas as pd

NVALID_MIN = 5

BELOW_CENSOR = -1e10

def to1d(u):
    u = np.atleast_1d(u).astype(float)
    if u.ndim != 1:
        raise ValueError(f"Expected 1d array, got ndim={u.ndim}.")
    return u


def to2d(uv):
    uv = np.atleast_2d(uv).astype(float)
    if uv.ndim != 2:
        raise ValueError(f"Expected 2d array, got ndim={uv.ndim}.")

    if uv.shape[1] != 2:
        uv = uv.T

    if uv.shape[1] != 2:
        raise ValueError(f"Expected shape[1]=2, got {uv.shape[1]}.")

    return uv


def data2array(data, ndim):
    data = np.array(data).astype(np.float64)
    if data.ndim != ndim:
        errmess = f"Expected {ndim}dim array, got ndim={data.ndim}."
        raise ValueError(errmess)
    nonans = (~np.isnan(data)).sum(axis=0)
    return data, nonans


def univariate2sorted(data):
    data, nval = data2array(data, 1)
    data_sorted = np.sort(data, axis=0)

    if nval < NVALID_MIN:
        errmess = f"Expected length of valid data>={NVALID_MIN}, got {nval}."
        raise ValueError(errmess)

    return data_sorted, nval


def univariate2cases(data, censor):
    data, nonans = data2array(data, 1)

    # We set the censor close to data min to avoid potential
    # problems with computing log cdf for the censor
    if censor is None or np.isnan(censor):
        censor = -1e10
    else:
        censor = max(np.float64(censor), np.nanmin(data) - 1e-10)

    is_miss = pd.isnull(data)
    is_obs = data >= censor
    is_cens = data < censor
    icases = pd.DataFrame({"i11": is_obs,
                           "i21": is_cens,
                           "i31": is_miss})
    nocens = is_obs.sum()
    if nocens < NVALID_MIN:
        errmess = f"Expected at least {NVALID_MIN} uncensored values, "\
                  + f"got {nocens}."
        raise ValueError(errmess)

    return icases, data, censor


def univariate2censored(data, censor):
    icases, data, censor = univariate2cases(data, censor)
    is_obs = icases.i11
    dobs = data[is_obs]
    ncens = icases.i21.sum()
    return data, dobs, ncens, censor


def bivariate2cases(data, ycensor, zcensor):
    ycases, ydata, ycensor = univariate2cases(data[:, 0], ycensor)
    y_obs = ycases.i11
    y_cens = ycases.i21
    y_miss = ycases.i31

    zcases, zdata, zcensor = univariate2cases(data[:, 1], zcensor)
    z_obs = zcases.i11
    z_cens = zcases.i21
    z_miss = zcases.i31

    icases = pd.DataFrame({"i11": y_obs & z_obs,
                           "i21": y_cens & z_obs,
                           "i31": y_miss & z_obs,
                           "i12": y_obs & z_cens,
                           "i22": y_cens & z_cens,
                           "i32": y_miss & z_cens,
                           "i13": y_obs & z_miss,
                           "i23": y_cens & z_miss,
                           "i33": y_miss & z_miss})

    if len(icases.i33) > 0:
        errmess = "Expected at least one variable to be valid."
        raise ValueError(errmess)

    return icases
