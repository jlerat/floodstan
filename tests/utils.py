import json, re, math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FTESTS = Path(__file__).resolve().parent

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_stationids(skip=5):
    fs = FTESTS / "data"
    stationids = []
    for ifile, f in enumerate(fs.glob("*_AMS.csv")):
        if not ifile % skip == 0:
            continue

        sid = re.sub("_.*", "", f.stem)
        if re.search("LIS", sid):
            continue
        stationids.append(sid)

    return stationids + ["hard"]


def get_ams(stationid):
    if stationid == "hard":
        fd = FTESTS / "data" / "LogPearson3_divergence_test.csv"
        y = pd.read_csv(fd).squeeze()
        y.index = np.arange(1990, 1990 + len(y))
        return y
    else:
        fs = FTESTS / "data" / f"{stationid}_AMS.csv"
        df = pd.read_csv(fs, skiprows=15, index_col=0)
        return df.iloc[:, 0]


def get_info():
    fs = FTESTS / "data" / "stations.csv"
    df = pd.read_csv(fs, skiprows=17)

    df.columns = [re.sub(" |,", "_", re.sub(" \\(.*", "", cn)) \
                            for cn in df.columns]
    df.loc[:, "Station_ID"] = df.Station_ID.astype(str)
    df = df.set_index("Station_ID")
    stationids = get_stationids()
    df = df.loc[stationids, :]

    return df


def add_gaussian_covariate(y):
    scale = np.nanstd(y) / 5
    z = y + np.random.normal(0, scale, size=len(y))

    z.iloc[-2] = np.nan # to add a missing data in z
    df = pd.DataFrame({"y": y, "z": z}).sort_index()
    return df.y, df.z


