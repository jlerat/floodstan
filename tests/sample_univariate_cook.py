import json, re, math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm, mvn, uniform
from scipy.stats import ks_1samp
from scipy.stats import percentileofscore

import warnings

from hydrodiy.plot import putils

from floodstan import marginals
from floodstan import sample
from floodstan import univariate_censored_sampling
from floodstan import report

from test_sample_univariate import get_stationids
from test_sample_univariate import get_ams


SEED = 5446

FTESTS = Path(__file__).resolve().parent

FLOGS = FTESTS / "logs" / "univariate_cook"
FLOGS.mkdir(exist_ok=True, parents=True)


def univariate_sampling_cook(marginal, stationid, debug):
    # Testing univariate sampling following the process described by
    # Samantha R Cook, Andrew Gelman & Donald B Rubin (2006)
    # Validation of Software for Bayesian Models Using Posterior Quantiles,
    # Journal of Computational and Graphical Statistics, 15:3, 675-692,
    # DOI: 10.1198/106186006X136976
    flog = FLOGS / f"univariate_{marginal}_{stationid}.log"
    LOGGER = sample.get_logger(level="INFO", stan_logger=False,
                               flog=flog)

    if debug and (stationid != "201001"):
        LOGGER.info(f"Debug mode - stationid={stationid}.. skip.")
        return

    LOGGER.info(f"Sampling {marginal}/{stationid}")

    # Stan config
    stan_nchains = 3 if debug else 10
    stan_nsamples = 1000 if debug else 10000
    stan_warmup = 1000 if debug else 10000

    # Large number of values to check we can get the "true" parameters
    # back from sampling
    nvalues = 50
    nrepeat = 10 if debug else 100

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
    yshape1_prior = [min(0.5, max(-0.5, dist.shape1)), 0.2]

    test_stat = []
    nsuccess = 0
    # .. 2 x number of tries to get nrepeat samples
    # .. in case of failure. We stop when we have nrepeat successes
    for repeat in range(2*nrepeat):
        desc = f"[{stationid}, {marginal}] test {nsuccess}/{nrepeat}"
        LOGGER.info(desc)

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
        fout = FTESTS / "sampling" / "univariate_cook" / stationid / marginal
        fout.mkdir(parents=True, exist_ok=True)
        for f in fout.glob("*.*"):
            f.unlink()

        # Sample
        try:
            smp = univariate_censored_sampling(data=stan_data,
                                               inits=stan_inits,
                                               chains=stan_nchains,
                                               seed=SEED,
                                               parallel_chains=1,
                                               iter_warmup=stan_warmup,
                                               iter_sampling=stan_nsamples//stan_nchains,
                                               output_dir=fout)
        except Exception as err:
            test_stat.append({"sucess": 1})
            continue

        # Get sample data
        df = smp.draws_pd()
        res = report.process_stan_diagnostic(smp.diagnose())

        # Clean
        for f in fout.glob("*.*"):
            f.unlink()

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

    LOGGER.info("Sampling done. Now plotting.")
    # Save
    test_stat = pd.DataFrame(test_stat)
    fr = FTESTS / "images" / "sampling" / "univariate_cook" /\
        f"univariate_sampling_{stationid}_{marginal}.csv"
    fr.parent.mkdir(exist_ok=True, parents=True)
    test_stat.to_csv(fr)

    # plot
    plt.close("all")
    axwidth, axheight = 6, 6
    fig, ax = plt.subplots(figsize=(axwidth, axheight), layout="tight")

    perc = test_stat.filter(regex="perc", axis=1)
    putils.ecdfplot(ax, perc)
    ax.legend(loc=4)
    ax.axline((0, 0), slope=1, linestyle="--", lw=0.9)

    nsuccess = (test_stat.success == 2).sum()
    ntotal = len(test_stat)
    title = f"{stationid}/{marginal} - {nsuccess}/{ntotal} samples"
    ax.set_title(title)

    # Report
    txt = f"{nsuccess} successes / {ntotal}\n\nStan diagnostics:\n"
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

    LOGGER.info("Process completed")


def main(marginal, debug):
    for stationid in get_stationids()[:3]:
        univariate_sampling_cook(marginal, stationid, debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run cook tests for univariate sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-m", "--marginal", help="Marginal name",
                        type=str, required=True,
                        choices=list(marginals.MARGINAL_NAMES.keys()))
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()
    marginal = args.marginal
    debug = args.debug

    main(marginal, debug)
