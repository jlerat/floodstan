import re
import numpy as np
import pandas as pd

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
DESIGN_ARIS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]


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
               obs_prefix="OBS"):
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

    Returns
    -------
    report_df : pandas.DataFrame
        Reported variables for all parameters

    report_stat : pandas.DataFrame
        Statistics for all reported variables.
        See floodstan.report.QUANTILES.
    """
    if truncated_probability < 0 or truncated_probability > 0.99:
        errmess = "Expected truncated probability in [0, 0.99]."
        raise ValueError(errmess)

    # Use marginal params if no params provided
    if params is None:
        params = pd.DataFrame([marginal.params])
        params.columns = ["locn", "logscale", "shape1"]

    # Prepare aris
    design_aris = np.array(design_aris)
    design_cdf = 1 - 1./design_aris
    # .. correct CDF with truncated probability
    design_cdf = (design_cdf - truncated_probability)\
        / (1 - truncated_probability)
    design_columns = [f"DESIGN_ARI{ari}" for ari in design_aris]

    # Prepare obs
    if observed is not None:
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

    # Detect parameter names
    params_columns = {}
    for pname in ["locn", "logscale", "shape1"]:
        cc = [cn for cn in params.columns if re.search(f"{pname}$", cn)]

        if len(cc) == 0:
            raise ValueError(f"No column in params refers to {pname}")

        if len(cc) > 1:
            errmess = f"More than one column in params refers to {pname}"
            raise ValueError(errmess)

        params_columns[pname] = cc[0]

    # Prepare numpy array to store set values
    ndesign = len(design_columns)
    nset = ndesign
    columns = design_columns
    if observed is not None:
        nobs = len(obs_columns_aep)
        nset += 2*nobs
        columns += obs_columns_aep + obs_columns_ari

    toset = np.zeros((len(params), nset))

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

        # .. compute design streamflow
        toset[iparams, :ndesign] = marginal.ppf(design_cdf)

        # .. compute aep of historical floods
        if observed is not None:
            cdf = truncated_probability\
                    + (1 - truncated_probability) * marginal.cdf(obs_values)
            toset[iparams, ndesign: ndesign + nobs] = (1. - cdf) * 100

            aris = 1. / (1. - cdf)
            toset[iparams, ndesign + nobs: ndesign + 2 * nobs] = aris

    report_df.loc[:, columns] = toset

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

    return report_stat, report_df
