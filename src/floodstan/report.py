import re
import numpy as np

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
        else:
            if delem == "divergence":
                prop_div = float(re.sub(".*\\(|%\\).*", "", line))
                if prop_div < 5:
                    stan_status[nelem] = "satisfactory"
                else:
                    stan_status[nelem] = line.strip()
            else:
                stan_status[nelem] = line.strip()

    return stan_status


def ams_report(marginal, params, observed=None,
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

    # Initialise report data
    report_df = params.copy()

    report_columns = []
    if observed is not None:
        report_columns += [f"{obs_prefix}{hkey}_AEP[%]"
                           for hkey, _ in observed.items()]
        report_columns += [f"{obs_prefix}{hkey}_ARI[yr]"
                           for hkey, _ in observed.items()]
    report_columns += [f"DESIGN_ARI{a}" for a in design_aris]
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

    # Loop through parameters
    for pidx, p in params.iterrows():
        # .. set parameters
        marginal.locn = p[params_columns["locn"]]
        marginal.logscale = p[params_columns["logscale"]]
        marginal.shape1 = p[params_columns["shape1"]]

        # .. compute aep of historical floods
        if observed is not None:
            for hkey, qh in observed.items():
                # .. correct CDF with truncated probability
                cdf = truncated_probability\
                    + (1 - truncated_probability) * marginal.cdf(qh)
                # .. store cdf
                cn = f"{obs_prefix}{hkey}_AEP[%]"
                report_df.loc[pidx, cn] = (1 - cdf) * 100

                cn = f"{obs_prefix}{hkey}_ARI[yr]"
                report_df.loc[pidx, cn] = 1. / (1 - cdf)

        # .. compute streamflow
        for ari in design_aris:
            cdf = 1 - 1./ari
            # .. correct CDF with truncated probability
            cdf = (cdf - truncated_probability) / (1 - truncated_probability)
            # .. compute design value
            report_df.loc[pidx, f"DESIGN_ARI{ari}"] = marginal.ppf(cdf)

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
