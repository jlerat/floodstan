import numpy as np
import pandas as pd


def get_start_end_of_water_year(year, water_year_start):
    start = pd.to_datetime(f"{year}-{water_year_start}-01")
    end = start+pd.DateOffset(years=1)-pd.DateOffset(days=1)
    return start, end


def get_annual_indexes(se, water_year_start):
    years = np.unique(se.index.year)
    annual_idx = pd.Series(-1, index=se.index)
    for year in years:
        start, end = get_start_end_of_water_year(year,
                                                 water_year_start)
        idx = (se.index >= start) & (se.index <= end)
        annual_idx[idx] = year

    return annual_idx


def get_annual_maximums(se, annual_idx,
                        water_year_start):
    df = []
    ievent = 1
    for year in np.sort(annual_idx.unique()):
        if year < 0:
            continue

        idx = year == annual_idx
        nvalyear = idx.sum()
        nvalid = se.loc[idx].notnull().sum()
        timepeak = pd.NaT
        eventid = "NA"
        peak = np.nan
        if nvalid > 0:
            nstdok = (se.loc[idx].std() > 1e-3).sum()
            if nstdok > 0:
                timepeak = se.loc[idx].idxmax()
                peak = se.loc[timepeak]
                eventid = f"F{ievent:03d}"
                ievent += 1

        start, end = get_start_end_of_water_year(year,
                                                 water_year_start)
        dd = {
            "EVENTID": eventid,
            "TIMEPEAK": timepeak,
            "PEAK": peak,
            "NVALYEAR": nvalyear,
            "NVALID": nvalid,
            "WATER_YEAR": year,
            "WATER_YEAR_START": start,
            "WATER_YEAR_END": end
            }
        df.append(dd)

    return pd.DataFrame(df)


def compute_ams(se, water_year_start="jan",
                gap_min=30, qmin_factor=0.75):
    """ Extract the Annual Maximum series (AMS) from timeseries.

    Parameters
    ----------
    se : pandas.core.Series
        Time series data.
    water_year_start : str
        Month marking the start of the water year.
    gap_min : int
        Minimum gap between two floods in days.

    Returns
    -------
    ams : pandas.Core.Series
        AMS series.
    """
    # compute first annual max
    annual_idx = get_annual_indexes(se, water_year_start)
    ams = get_annual_maximums(se, annual_idx, water_year_start)

    # Check when events are too close
    # If yes, adjust annual_idx to avoid splitting events
    delta = ams.TIMEPEAK.diff().dt.days
    tooshort = delta < gap_min
    if tooshort.sum() > 0:
        for ievent in delta.index[tooshort & (delta.index > 0)]:
            start, end = ams.TIMEPEAK[[ievent-1, ievent]]
            qq = ams.PEAK[[ievent-1, ievent]]
            if qq.isnull().any():
                continue

            yy = ams.WATER_YEAR[[ievent-1, ievent]]

            # Qmin test not passed. We have to join the events
            imax = qq.idxmax()
            start = start-pd.DateOffset(days=gap_min//2)
            end = end+pd.DateOffset(days=gap_min//2)
            annual_idx.loc[start:end] = yy[imax]

    return get_annual_maximums(se, annual_idx,
                               water_year_start)
