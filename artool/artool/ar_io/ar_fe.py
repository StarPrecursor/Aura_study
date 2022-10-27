import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def get_dt_features(df, name, categories=["hour", "day", "week"], unit="us"):
    t = df[name]
    # check if t is a datetime object
    if not isinstance(t, pd.Timestamp):
        t = pd.to_datetime(t, unit=unit)
    df_out = pd.DataFrame()
    if "year" in categories:
        df_out[f"{name}__year"] = t.dt.year
        df_out[f"{name}__is_year_start"] = t.dt.is_year_start
        df_out[f"{name}__is_year_end"] = t.dt.is_year_end
        df_out[f"{name}__is_leap_year"] = t.dt.is_leap_year
    if "quarter" in categories:
        df_out[f"{name}__quarter"] = t.dt.quarter
        df_out[f"{name}__is_quarter_start"] = t.dt.is_quarter_start
        df_out[f"{name}__is_quarter_end"] = t.dt.is_quarter_end
    if "month" in categories:
        df_out[f"{name}__month"] = t.dt.month
        df_out[f"{name}__is_month_start"] = t.dt.is_month_start
        df_out[f"{name}__is_month_end"] = t.dt.is_month_end
    if "week" in categories:
        df_out[f"{name}__week"] = t.dt.isocalendar().week.astype(int)
    if "day" in categories:
        df_out[f"{name}__day"] = t.dt.day
        df_out[f"{name}__dayofweek"] = t.dt.dayofweek
        df_out[f"{name}__dayofyear"] = t.dt.dayofyear
        df_out[f"{name}__days_in_month"] = t.dt.days_in_month
        df_out[f"{name}__weekday"] = t.dt.weekday
        df_out[f"{name}__is_weekend"] = t.dt.dayofweek >= 5
        df_out[f"{name}__is_week_start"] = t.dt.dayofweek == 0
        df_out[f"{name}__is_week_end"] = t.dt.dayofweek >= 5
        cld = USFederalHolidayCalendar()
        holidays = cld.holidays(start=t.min(), end=t.max())
        df_out[f"{name}__is_holiday"] = t.isin(holidays)
    if "hour" in categories:
        df_out[f"{name}__hour"] = t.dt.hour
    if "minute" in categories:
        df_out[f"{name}__minute"] = t.dt.minute
    if "second" in categories:
        df_out[f"{name}__second"] = t.dt.second
    return df_out


def get_lag_features(df, name, lags=[1, 3, 5]):
    x = df[name]
    df_out = pd.DataFrame()
    for lag in lags:
        df_out[f"{name}__lag{lag}"] = x.shift(lag)
    return df_out


def get_diff_features(df, names, ref, suffix=""):
    df_out = pd.DataFrame()
    for name in names:
        df_out[f"{name}__diff{suffix}"] = df[name] - df[ref]
    return df_out


def get_rolling_features(df, names, lookbacks):
    df_out = pd.DataFrame()
    for lookback in lookbacks:
        for name in names:
            df_out[f"{name}__rol_mean"] = df[name].rolling(lookback).mean()
            df_out[f"{name}__rol_std"] = df[name].rolling(lookback).std()
            df_out[f"{name}__rol_min"] = df[name].rolling(lookback).min()
            df_out[f"{name}__rol_max"] = df[name].rolling(lookback).max()
            df_out[f"{name}__rol_skew"] = df[name].rolling(lookback).skew()
            df_out[f"{name}__rol_kurt"] = df[name].rolling(lookback).kurt()
    return df_out


def get_ewm_features(df, names, spans, scale=1, suffix=""):
    df_out = pd.DataFrame()
    for hl in spans:
        for name in names:
            df_out[f"{name}__ewm{hl}{suffix}"] = (
                df[name].ewm(halflife=hl * scale).mean()
            )
    return df_out


def get_delta_features(df, names, step=1):
    df_out = pd.DataFrame()
    for name in names:
        df_out[f"{name}__delta{step}"] = df[name] - df[name].shift(step)
    return df_out


def get_future_features(df, names, lookforwards, scale=1):
    df_out = pd.DataFrame()
    for lookforward in lookforwards:
        lf_value = lookforward * scale
        for name in names:
            df_out[f"{name}__future_{lookforward}"] = (
                df[name].rolling(lf_value).mean().shift(-lf_value)
            )
    return df_out
