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
    df_collect = {}
    for member in within:
        df_tmp = df[df[on] == member]
        df_collect[member] = df_tmp[feature_map.keys()].rename(columns=feature_map)
    return df_collect
