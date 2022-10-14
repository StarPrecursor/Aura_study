import re
from pathlib import Path

from typing import Union

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
