import json, re, math
import shutil
from pathlib import Path
from itertools import product as prod

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn, uniform
from scipy.stats import ks_1samp
from scipy.stats import percentileofscore

import pytest
import warnings

from hydrodiy.plot import putils

import importlib
from tqdm import tqdm

from floodstan import marginals
from floodstan import sample
from floodstan import univariate_censored_sampling
from floodstan import report

from test_sample_univariate import get_stationids
from test_sample_univariate import get_ams


SEED = 5446

FTESTS = Path(__file__).resolve().parent

@pytest.mark.cook
@pytest.mark.parametrize("marginal",
                         list(sample.MARGINAL_NAMES.keys()))
@pytest.mark.parametrize("stationid",
                         get_stationids()[:1])
def test_univariate_sampling(marginal, stationid, allclose):
    # Testing univariate sampling following the process described by
    # Samantha R Cook, Andrew Gelman & Donald B Rubin (2006)
    # Validation of Software for Bayesian Models Using Posterior Quantiles,
    # Journal of Computational and Graphical Statistics, 15:3, 675-692,
    # DOI: 10.1198/106186006X136976
    if marginal in ["Normal", "LogNormal", "GeneralizedPareto"]:
        pytest.skip(f"marginal {marginal} is not challenging. Skip.")

    # Stan config
    stan_nchains = 5
    stan_nsamples = 5000
    stan_warmup = 5000

    LOGGER = sample.get_logger(level="INFO", stan_logger=False)

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 50
    nrepeat = 30

    print("\n")

    y = get_ams(stationid)
    N = len(y)

    # Setup marginal
    if marginal in ["Gumbel", "LogNormal", "Normal", "Gamma"]:
        parnames = ["locn", "logscale"]
    else:
        parnames = ["locn", "logscale", "shape1"]

    dist = marginals.factory(marginal)
    dist.params_guess(y)

    if dist.shape1<marginals.SHAPE1_LOWER \
            or dist.shape1>marginals.SHAPE1_UPPER:
        pytest.skip("Shape parameter outside of accepted bounds.")

    # Prior distribution centered around dist params
    ylocn_prior = [dist.locn, abs(dist.locn)*0.5]
    ylogscale_prior = [dist.logscale, abs(dist.logscale)*0.5]
    yshape1_prior = [max(0.1, dist.shape1), 0.2]

    test_stat = []
    nsuccess = 0
    # .. 2 x number of tries to get nrepeat samples
    # .. in case of failure. We stop when we have nrepeat successes
    for repeat in range(2*nrepeat):
        desc = f"[{stationid}, {marginal}] test {nsuccess}/{nrepeat}"
        print(desc)

        # Generate parameters from prior
        try:
            dist.locn = norm.rvs(*ylocn_prior)
            dist.logscale = norm.rvs(*ylogscale_prior)
            dist.shape1 = norm.rvs(*yshape1_prior)
        except:
            continue

        # Generate data from prior params
        ysmp = dist.rvs(N)

        # Configure stan data and initialisation
        try:
            sv = sample.StanSamplingVariable(ysmp, marginal,
                                             ninits=stan_nchains)
        except:
            test_stat.append({"sucess": 0})
            continue

        sv.locn_prior = ylocn_prior
        sv.logscale_prior = ylogscale_prior
        sv.shape1_prior = yshape1_prior

        stan_data = sv.to_dict()
        stan_inits = sv.initial_parameters

        # Clean output folder
        fout = FTESTS / "sampling" / "univariate" / stationid / marginal
        fout.mkdir(parents=True, exist_ok=True)
        for f in fout.glob("*.*"):
            f.unlink()

        # Sample
        try:
            smp = univariate_censored_sampling(data=stan_data,
                                               inits=stan_inits,
                                               chains=stan_nchains,
                                               seed=SEED,
                                               iter_warmup=stan_warmup,
                                               iter_sampling=stan_nsamples//stan_nchains,
                                               output_dir=fout)
        except Exception as err:
            test_stat.append({"sucess": 1})
            continue

        # Get sample data
        df = smp.draws_pd()
        res = report.process_stan_diagnostic(smp.diagnose())

        # Test statistic
        Nsmp = len(df)
        res["success"] = 2
        for cn in parnames:
            th = dist[cn]
            val = df.loc[:, f"y{cn}"]
            res[f"{cn}_theoretical"] = th
            res[f"{cn}_values_mean"] = val.mean()
            res[f"{cn}_values_std"] = val.std()
            prc = percentileofscore(val, dist[cn]) / 100
            res[f"{cn}_perc"] = prc

        test_stat.append(res)

        # Stop iterating when the number of samples is met
        nsuccess += 1
        if nsuccess == nrepeat:
            break

    # Save
    test_stat = pd.DataFrame(test_stat)
    fr = FTESTS / "images" / f"univariate_sampling_{stationid}_{marginal}.csv"
    fr.parent.mkdir(exist_ok=True)
    test_stat.to_csv(fr)

    # plot
    plt.close("all")
    axwidth, axheight = 6, 6
    fig, ax = plt.subplots(figsize=(axwidth, axheight), layout="tight")

    perc = test_stat.filter(regex="perc", axis=1)
    putils.ecdfplot(ax, perc)
    ax.legend(loc=4)
    ax.axline((0, 0), slope=1, linestyle="--", lw=0.9)

    title = f"{stationid}/{marginal} - {len(test_stat)} samples / {nrepeat}"
    ax.set_title(title)

    # Report
    nsuccess = (test_stat.success == 2).sum()
    txt = f"{nsuccess} successes / {nrepeat}\n\nStan diagnostics:\n"
    for cn in ["treedepth", "divergence", "ebfmi", "effsamplesz", "rhat"]:
        prc = (test_stat.loc[:, cn] == "satisfactory").sum() / nsuccess
        prc *= 100
        txt += " "*4 + f"{cn:12s}: {prc:0.0f}% satis.\n"

    txt += "\n\nKS p-values:\n"
    color = "darkgreen"
    weight = "normal"
    for cn in parnames:
        x = test_stat.loc[:, f"{cn}_perc"]
        k = ks_1samp(x, uniform.cdf)
        txt += " "*4 + f"{cn:12s}: {k.pvalue:0.2f}\n"
        if k.pvalue < 0.05:
            color = "tab:red"
            weight = "bold"

    ax.text(0.02, 0.98, txt, va="top", ha="left",
            fontsize="large", transform=ax.transAxes,
            color=color, fontweight=weight)

    fp = fr.parent / f"{fr.stem}.png"
    fig.savefig(fp)
