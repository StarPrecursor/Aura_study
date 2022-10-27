import pandas as pd


def get_dt_features(df, name, categories=["hour", "day", "week"], unit="us"):
    t = df[name]
    # check if t is a datetime object
    if not isinstance(t, pd.Timestamp):
        t = pd.to_datetime(t, unit=unit)
    df_out = pd.DataFrame()
    if "year" in categories:
        df_out[f"{name}__year"] = t.year
        df_out[f"{name}__is_year_start"] = t.is_year_start
        df_out[f"{name}__is_year_end"] = t.is_year_end
        df_out[f"{name}__is_leap_year"] = t.is_leap_year
    if "quarter" in categories:
        df_out[f"{name}__quarter"] = t.quarter
        df_out[f"{name}__is_quarter_start"] = t.is_quarter_start
        df_out[f"{name}__is_quarter_end"] = t.is_quarter_end
    if "month" in categories:
        df_out[f"{name}__month"] = t.month
        df_out[f"{name}__is_month_start"] = t.is_month_start
        df_out[f"{name}__is_month_end"] = t.is_month_end
    if "week" in categories:
        df_out[f"{name}__week"] = t.week
        df_out[f"{name}__is_week_start"] = t.is_week_start
        df_out[f"{name}__is_week_end"] = t.dayofweek >= 5
    if "day" in categories:
        df_out[f"{name}__day"] = t.day
        df_out[f"{name}__dayofweek"] = t.dayofweek
        df_out[f"{name}__dayofmonth"] = t.dayofmonth
        df_out[f"{name}__dayofyear"] = t.dayofyear
        df_out[f"{name}__days_in_month"] = t.days_in_month
        df_out[f"{name}__days_in_year"] = t.days_in_year
        df_out[f"{name}__weekday"] = t.weekday
        df_out[f"{name}__is_weekend"] = t.dayofweek >= 5
    if "hour" in categories:
        df_out[f"{name}__hour"] = t.hour
    if "minute" in categories:
        df_out[f"{name}__minute"] = t.minute
    if "second" in categories:
        df_out[f"{name}__second"] = t.second
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
