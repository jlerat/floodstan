import re
import numpy as np
import pandas as pd

from floodstan.marginals import PARAMETERS
from floodstan import quadapprox
from floodstan.freqplots import cdf_to_reduced_variate
from floodstan.freqplots import reduced_variate_to_cdf


QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
DESIGN_ARIS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

# This number of approximation nodes ensures
# an accuracy of quantile function lower than 5e-3
# (see test_quadapprox.test_approx_cdf)
N_POSTPRED_APPROX = 2000

# CDF range covered in posterior predictive approx
CDF_APPROX_MIN = 0.3
CDF_APPROX_MAX = 1 - 1e-10


def _prepare_design_aris(design_aris, truncated_probability):
    if truncated_probability < 0 or truncated_probability > 0.99:
        errmess = "Expected truncated probability in [0, 0.99]."
        raise ValueError(errmess)

    design_aris = np.array(design_aris)
    design_cdf = 1 - 1./design_aris
    # Correct CDF with truncated probability
    design_cdf = (design_cdf - truncated_probability)\
        / (1 - truncated_probability)
    design_columns = [f"DESIGN_ARI{ari}" for ari in design_aris]

    # Approx nodes to compute posterior predictive distribution
    # .. regular spacing
    c0 = CDF_APPROX_MIN
    c1 = CDF_APPROX_MAX

    if design_cdf.min() < c0:
        errmess = f"Expected design cdf > {c0:3.3e}."
        raise ValueError(errmess)

    if design_cdf.max() > c1:
        errmess = f"Expected design cdf < {c1:3.3e}."
        raise ValueError(errmess)

    post_pred_cdf = np.linspace(c0, c1, N_POSTPRED_APPROX // 2)

    # .. Gumbel spacing
    x0 = cdf_to_reduced_variate(c0, "gumbel")
    x1 = cdf_to_reduced_variate(c1, "gumbel")
    xx = np.linspace(x0, x1, N_POSTPRED_APPROX // 2)
    pp = reduced_variate_to_cdf(xx, "gumbel")
    post_pred_cdf = np.unique(np.sort(np.concatenate([post_pred_cdf, pp])))

    return design_aris, design_cdf, design_columns, post_pred_cdf


def _detect_params_columns(params):
    params_columns = {}
    for pname in PARAMETERS:
        cc = [cn for cn in params.columns if re.search(f"{pname}$", cn)]

        if len(cc) == 0:
            raise ValueError(f"No column in params refers to {pname}")

        if len(cc) > 1:
            errmess = f"More than one column in params refers to {pname}"
            raise ValueError(errmess)

        params_columns[pname] = cc[0]
    return params_columns


def process_stan_diagnostic(diag):
    """ Analyse stan diagnostic data.

    Parameters
    ----------
    diag : str
        Stan diagnostic generated from

    Returns
    -------
    stan_status : dict
        Stan diagnostic metrics.
    """
    diag_pat = "Checking|Processing|consider|consider"\
        + "|incomplete mixing|prematurely"\
        + "|not fully able|Try|try"
    rep_pat = ".*parameters had"
    diag = re.sub(":\n ", ": ", diag)
    diag = [re.sub(rep_pat, "", li) for li in diag.split("\n")
            if li != "" and not re.search(diag_pat, li)]

    stan_status = dict(message=" ".join(diag))
    patterns = {
        "treedepth": "(T|t)reedepth",
        "divergence": "divergen(t|ce)",
        "ebfmi": "E-BFMI",
        "effsamplesz": "Effective|effective draws",
        "rhat": "R-hat"
        }

    for delem, pat in patterns.items():
        nelem = f"{delem}"
        line = [li for li in diag if re.search(pat, li)]
        if len(line) == 0:
            stan_status[nelem] = "unknown"
            continue

        line = line[0]
        if re.search("satisfactory|No divergent", line):
            stan_status[nelem] = "satisfactory"
            if delem == "divergence":
                stan_status[f"{nelem}_proportion"] = 0.
        else:
            if delem == "divergence":
                prop_div = float(re.sub(".*\\(|%\\).*", "", line))
                stan_status[f"{nelem}_proportion"] = prop_div
                if prop_div < 5:
                    stan_status[nelem] = "satisfactory"
                else:
                    stan_status[nelem] = line.strip()
            else:
                stan_status[nelem] = line.strip()

    return stan_status


def ams_report(marginal, params=None, observed=None,
               truncated_probability=0,
               design_aris=DESIGN_ARIS,
               obs_prefix="OBS",
               posterior_predictive=True):
    """ Generate report variables.

    Parameters
    ----------
    marginal : floodstan.marginals.FloodFreqDistribution
        Flood frequency distribution used.
    params : pandas.DataFrame
        List of parameter sets
    observed : dict
        Series of observed values for which we whish to
        know the frequency characteristics.
    truncated_probability : float
        Probability of truncated values.
    design_aris : list
        List of design flood ari to be computed.
    obs_prefix : str
        Prefix appended before observed variables.
    posterior_predictive : bool
        Compute quantiles from posterior predictive
        distribution.

    Returns
    -------
    report_df : pandas.DataFrame
        Reported variables for all parameters

    report_stat : pandas.DataFrame
        Statistics for all reported variables.
        See floodstan.report.QUANTILES.
    """
    design_aris, design_cdf, design_columns, post_pred_cdf = \
        _prepare_design_aris(design_aris, truncated_probability)

    has_obs = observed is not None

    # Use marginal params if no params provided
    if params is None:
        params = pd.DataFrame([marginal.params])
        params.columns = ["locn", "logscale", "shape1"]

    params_columns = _detect_params_columns(params)

    # Prepare obs
    if has_obs:
        obs_idx = list(observed.keys())
        obs_values = np.array([observed[k] for k in obs_idx])
        obs_columns_aep = [f"{obs_prefix}{o}_AEP[%]" for o in obs_idx]
        obs_columns_ari = [f"{obs_prefix}{o}_ARI[yr]" for o in obs_idx]

    # Initialise report data
    report_df = params.copy()

    report_columns = []
    if observed is not None:
        report_columns += [f"{obs_prefix}{hkey}_AEP[%]"
                           for hkey, _ in observed.items()]
        report_columns += [f"{obs_prefix}{hkey}_ARI[yr]"
                           for hkey, _ in observed.items()]
    report_columns += design_columns
    report_df.loc[:, report_columns] = np.nan

    # Prepare numpy array to store set values
    ndesign = len(design_columns)
    nset = ndesign
    columns = design_columns
    if has_obs:
        nobs = len(obs_columns_aep)
        nset += 2*nobs
        columns += obs_columns_aep + obs_columns_ari

    toset = np.zeros((len(params), nset))

    # prepare data for posterior predictive distribution
    # computation
    means = params.mean()
    marginal.locn = means.loc[params_columns["locn"]]
    marginal.logscale = means.loc[params_columns["logscale"]]
    if marginal.has_shape:
        marginal.shape1 = means.loc[params_columns["shape1"]]

    xi = np.unique(marginal.ppf(post_pred_cdf))
    xm = (xi[:-1] + xi[1:]) / 2

    # .. compute quantile distribution using mean params
    #    of design floods
    design_meanp = marginal.ppf(design_cdf)

    if has_obs:
        cdf = marginal.cdf(obs_values)
        cdf = truncated_probability + (1 - truncated_probability) * cdf
        obs_aep_meanp = (1 - cdf) * 100
        obs_ari_meanp = 1 / (1 - cdf)

    # .. initialise parameter vectors
    nxi = len(xi) + 1
    a_coefs = np.zeros(nxi)
    b_coefs = np.zeros(nxi)
    c_coefs = np.zeros(nxi)
    nparams_ok = 0.

    # Loop through parameters
    for iparams, (_, p) in enumerate(params.iterrows()):
        # .. set parameters
        try:
            marginal.locn = p[params_columns["locn"]]
            marginal.logscale = p[params_columns["logscale"]]
            if marginal.has_shape:
                marginal.shape1 = p[params_columns["shape1"]]
        except ValueError:
            continue

        nparams_ok += 1

        # .. get quadratic aprox coefficient to compute predictive
        #    posterior distribution
        if posterior_predictive:
            fi = marginal.cdf(xi)
            fm = marginal.cdf(xm)
            a, b, c = quadapprox.get_coefficients(xi, fi, fm)

            a_coefs += a
            b_coefs += b
            c_coefs += c

        # .. compute design streamflow
        toset[iparams, :ndesign] = marginal.ppf(design_cdf)

        # .. compute aep of historical floods
        if observed is not None:
            cdf = truncated_probability\
                    + (1 - truncated_probability) * marginal.cdf(obs_values)
            toset[iparams, ndesign: ndesign + nobs] = (1. - cdf) * 100

            aris = 1. / (1. - cdf)
            toset[iparams, ndesign + nobs: ndesign + 2 * nobs] = aris

    # Set values in report
    report_df.loc[:, columns] = toset

    # Standardize coefs of posterior predictive
    if posterior_predictive:
        a_coefs = a_coefs / nparams_ok
        b_coefs = b_coefs / nparams_ok
        c_coefs = c_coefs / nparams_ok

    # Build stat report
    # .. compute stat
    cp = [v for k, v in params_columns.items()]
    cc = [cn for cn in report_df.columns
          if cn in report_columns or cn in cp]
    report_stat = report_df.loc[:, cc].describe(percentiles=QUANTILES)
    report_stat = report_stat.drop(["count"], axis=0)

    # .. compute confidence interval
    ridx = report_stat.index
    if "5%" in ridx and "95%" in ridx:
        report_stat.loc["CI90", :] = report_stat.loc["95%"]\
            - report_stat.loc["5%"]

    # .. compute finite value proportion
    cc = report_stat.columns
    report_stat.loc["ISFINITE[%]", cc] = report_df.loc[:, cc]\
        .apply(lambda x: np.isfinite(x).sum() / len(x) * 100)
    report_stat.loc["ISZERO[%]", cc] = report_df.loc[:, cc]\
        .apply(lambda x: (np.abs(x) < 1e-10).sum() / len(x) * 100)

    # .. final formatting
    report_stat = report_stat.T
    report_stat.columns = [cn.upper() for cn in report_stat.columns]

    # .. compute posterior predictive distribution
    if posterior_predictive:
        # Design quantiles
        design_post = quadapprox.inverse(design_cdf, xi, a_coefs, b_coefs, c_coefs)

        # Obs AEP
        if has_obs:
            cdf = quadapprox.forward(obs_values, xi, a_coefs, b_coefs, c_coefs)
            cdf = truncated_probability + (1 - truncated_probability) * cdf
            obs_aep_post = (1 - cdf) * 100
            obs_ari_post = 1 / (1 - cdf)
    else:
        design_post = np.nan * np.zeros_like(design_cdf)

    cnp = "POSTERIOR_PREDICTIVE"
    cnm = "EXPECTED_PARAMETERS"
    for cn in [cnp, cnm]:
        report_stat.loc[:, cn] = np.nan

    for ari, qp, qm in zip(design_aris, design_post, design_meanp):
        idx = f"DESIGN_ARI{ari}"
        report_stat.loc[idx, cnp] = qp
        report_stat.loc[idx, cnm] = qm

    if has_obs:
        report_stat.loc[obs_columns_aep, cnm] = obs_aep_meanp
        report_stat.loc[obs_columns_ari, cnm] = obs_ari_meanp

        if posterior_predictive:
            report_stat.loc[obs_columns_aep, cnp] = obs_aep_post
            report_stat.loc[obs_columns_ari, cnp] = obs_ari_post

    # .. add expected parameters
    idx = [f"y{n}" for n in PARAMETERS]
    report_stat.loc[idx, cnm] = means.loc[idx]

    return report_stat, report_df
