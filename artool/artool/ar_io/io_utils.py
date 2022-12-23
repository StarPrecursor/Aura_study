import re
from pathlib import Path
from typing import Union

import git
import pandas as pd


# fetch names with given regex
def fetch_names(
    dir: Path, regex: str, sort: bool = True, group_id: Union[int, list] = 1
) -> list:
    if isinstance(group_id, int):
        group_id = [group_id]
    # get matched groups
    names = []
    for p in dir.iterdir():
        mt = re.match(regex, p.name)
        if mt:
            names.append(mt.groups(*group_id))
    if sort:
        names.sort()
    return names


def get_git_root(cur) -> Path:
    cur = Path(cur)
    git_repo = git.Repo(cur, search_parent_directories=True)
    git_dir = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_dir)


def get_git_rel_path(cur) -> str:
    cur = Path(cur)
    git_dir = get_git_root(cur)
    return cur.relative_to(git_dir).as_posix()


def decouple_df(df, on="symbol", within=[], feature_map={}):
    """Decouple a dataframe into multiple dataframes.

    - feature_map: dict, {feature_name_out: feature_name_in}

    """
    df_collect = {}
    rename_dict = {v: k for k, v in feature_map.items()}
    for member in within:
        df_tmp = df[df[on] == member]
        df_collect[member] = df_tmp[rename_dict.keys()].rename(columns=rename_dict)
    return df_collect


def select_df_date_range(df, time_col, start_date=None, end_date=None):
    if start_date is None:
        start_date = df[time_col].min()
    if end_date is None:
        end_date = df[time_col].max()
    return df[
        (pd.to_datetime(df[time_col]) >= start_date)
        & (pd.to_datetime(df[time_col]) <= end_date)
    ]
