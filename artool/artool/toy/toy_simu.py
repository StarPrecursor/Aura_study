import numpy as np


def get_pnl_simple(x_sig, funding, buythres, sellthres, feeRate=3e-4):
    # calculate profit and loss, fixed volume of trade to 1
    obsNum = len(x_sig)
    signal = np.zeros(obsNum)
    signal[x_sig > buythres] = 1
    signal[x_sig < sellthres] = -1
    position = np.zeros(obsNum)
    fee = np.zeros(obsNum)
    toBuy = False
    toSell = False
    for ii in range(obsNum):
        if ii >= 1:
            position[ii] = position[ii - 1]
        if toBuy:
            position[ii] = 1
            fee[ii] = feeRate
        if toSell:
            position[ii] = 0
            fee[ii] = feeRate
        toBuy = False
        toSell = False
        if signal[ii] == 1 and position[ii] == 0:
            toBuy = True
        if signal[ii] == -1 and position[ii] == 1:
            toSell = True
    pnl = np.sum(position * funding) - np.sum(fee)
    return pnl
