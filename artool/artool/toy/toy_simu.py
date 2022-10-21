import numpy as np


def get_pnl_simple(x_sig, funding, buythres, sellthres, feeRate=3e-4, verbose=False):
    """Calculate profit and loss for single symbol
    - fixed volume of trade to 1
    - only short futures
    """
    n = len(x_sig)
    signal = np.zeros(n)
    signal[x_sig > buythres] = 1
    signal[x_sig < sellthres] = -1
    # Simulation
    pos_short = np.zeros(n)
    fee = np.zeros(n)
    go_short = False
    out_short = False
    for i in range(n):
        # record trading
        pos_short[i] = pos_short[i - 1]
        if go_short:
            pos_short[i] = 1
            fee[i] = feeRate
        if out_short:
            pos_short[i] = 0
            fee[i] = feeRate
        # update flag
        go_short = False
        out_short = False
        if signal[i] == 1 and pos_short[i] == 0:
            go_short = True
        if signal[i] == -1 and pos_short[i] == 1:
            out_short = True
    # Show non-zero positions
    if verbose:
        # conunt non-zero
        n_short = np.sum(pos_short != 0)
        print(f"Position len {n}")
        print(f"Number of non-zero short: {n_short}")
    # Calculate PnL
    pnl = np.sum(pos_short * funding) - np.sum(fee)
    return pnl


def get_pnl_2side(
    x_sig,
    funding,
    buythres_short,
    sellthres_short,
    buythres_long,
    sellthres_long,
    feeRate=3e-4,
    verbose=False,
):
    """Calculate profit and loss
    - fixed volume of trade to 1
    - short and long futures
    """
    n = len(x_sig)
    sig_short = np.zeros(n)
    sig_short[x_sig > buythres_short] = 1
    sig_short[x_sig < sellthres_short] = -1
    sig_long = np.zeros(n)
    sig_long[x_sig < buythres_long] = 1
    sig_long[x_sig > sellthres_long] = -1
    # Simulation
    pos_short = np.zeros(n)
    pos_long = np.zeros(n)
    fee = np.zeros(n)
    go_short = False
    out_short = False
    go_long = False
    out_long = False
    for i in range(n):
        # update short
        pos_short[i] = pos_short[i - 1]
        if go_short:
            pos_short[i] = 1
            fee[i] += feeRate
        if out_short:
            pos_short[i] = 0
            fee[i] += feeRate
        go_short = sig_short[i] == 1 and pos_short[i] == 0
        out_short = sig_short[i] == -1 and pos_short[i] == 1
        # update long
        pos_long[i] = pos_long[i - 1]
        if go_long:
            pos_long[i] = -1
            fee[i] += feeRate
        if out_long:
            pos_long[i] = 0
            fee[i] += feeRate
        go_long = sig_long[i] == 1 and pos_long[i] == 0
        out_long = sig_long[i] == -1 and pos_long[i] == -1
    # Show non-zero positions
    if verbose:
        # conunt non-zero
        n_short = np.sum(pos_short != 0)
        n_long = np.sum(pos_long != 0)
        print(f"Position len {n}")
        print(f"Number of non-zero short: {n_short}")
        print(f"Number of non-zero long: {n_long}")
    # Calculate PnL
    pnl = np.sum(pos_short * funding) + np.sum(pos_long * funding) - np.sum(fee)
    pnl /= 2
    return pnl


def get_pnl_1side_dynamic(x_sig, funding, buythres, sellthres, max_holds, feeRate=3e-4):
    """Calculate profit and loss for single symbol
    - fixed volume of trade to 1
    - only short futures

    x_sig: signal variable of all symbols, each row represents a trade time, each column represents a symbol

    """
    n = len(x_sig)
    x_shape = x_sig.shape
    signal = np.zeros(x_shape)
    signal[x_sig > buythres] = 1
    signal[x_sig < sellthres] = -1
    # Pad one row of zeros to x_sig
    x_sig = np.vstack((np.zeros(x_shape[1]), x_sig))
    # Simulation
    pos_short = np.zeros(x_shape)
    fee = np.zeros(x_shape)
    go_short = np.zeros(x_shape[1], dtype=bool)
    out_short = np.zeros(x_shape[1], dtype=bool)
    cur_holds = 0
    for i in range(n):
        # get out of positions based on out_short
        pos_short[i] = pos_short[i - 1]
        pos_short[i][out_short] = 0
        fee[i][out_short] += feeRate
        cur_holds = np.sum(pos_short[i] != 0)
        remain_holds = max_holds - cur_holds
        # get into positions based on go_short
        if remain_holds == 0:
            pass
        else:
            if np.sum(go_short) <= remain_holds:
                pos_short[i][go_short] = 1
                fee[i][go_short] += feeRate
            else:
                # select with max x_sig values
                go_short_idx = np.argsort(x_sig[i-1])[::-1][:remain_holds]
                pos_short[i][go_short_idx] = 1
                fee[i][go_short_idx] += feeRate
                # randomly select
                #go_short_idx = np.where(go_short)[0]
                #np.random.shuffle(go_short_idx)
                #go_short_idx = go_short_idx[:remain_holds]
                #pos_short[i][go_short_idx] = 1
                #fee[i][go_short_idx] += feeRate
        # update flag
        go_short = (signal[i] == 1) & (pos_short[i] == 0)
        out_short = (signal[i] == -1) & (pos_short[i] == 1)
    # Calculate PnL
    pnl = np.sum(pos_short * funding) - np.sum(fee)
    return pnl


