import numpy as np
from scipy.special import factorial as npfactorial
import os
import math


models = ["GM", "NB2", "DW2", "DW3", "S", "TL", "IFRSB", "IFRGSB"]

def hazard_numerical(model, i, args):
    if model == "GM":
        f = args[0]
        return f
    elif model == "DW3":
        f = 1 - math.exp(-args[0] * i**args[1])
        return f
    elif model == "DW2":
        f = 1 - args[0]**(i**2 - (i - 1)**2)
        return f
    elif model == "IFRGSB":
        f = 1 - args[0] / ((i - 1) * args[1] + 1)
        return f
    elif model == "IFRSB":
        f = 1 - args[0] / i
        return f
    elif model == "NB2":
        f = (i * args[0]**2)/(1 + args[0] * (i - 1))
        return f
    elif model == "S":
        f = args[0] * (1 - args[1]**i)
        return f
    elif model == "TL":
        try:
            f = (1 - math.exp(-1/args[1])) / (1 + math.exp(- (i - args[0])/args[1]))
        except OverflowError:
            f = float('inf')
        return f



