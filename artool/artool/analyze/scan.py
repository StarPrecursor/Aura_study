import pandas as pd
import numpy as np
import tqdm
import logging

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("artool")

def grid_scan(dimensions, func, **kwargs):
    """
    Grid scan over a set of dimensions.

    Parameters
    ----------
    dimensions : dict
        Dictionary of dimension names and values.
    func : callable
        Function to call for each point in the grid.
    kwargs : dict
        Keyword arguments to pass to func.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with results of the scan.
    """
    # Get all combinations of dimensions
    dim_names = list(dimensions.keys())
    dim_values = list(dimensions.values())
    dim_combos = np.array(np.meshgrid(*dim_values)).T.reshape(-1, len(dim_values))
    # Run the function for each combination
    results = {k: [] for k in dim_names}
    results["score"] = []

    # Scan with multiprocessing
    logger.info("Performing grid scan.")
    with ProcessPoolExecutor() as executor:
        futures = []
        for dim_combo in dim_combos:
            futures.append(
                executor.submit(
                    func,
                    **{k: v for k, v in zip(dim_names, dim_combo)},
                    **kwargs,
                )
            )
            for k, v in zip(dim_names, dim_combo):
                results[k].append(v)
        for f in tqdm.tqdm(futures):
            results["score"].append(f.result())

    # Create a DataFrame with the results
    df = pd.DataFrame(results)
    return df
