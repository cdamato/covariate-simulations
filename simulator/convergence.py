import os
import ctypes
import numpy as np
import math
import pandas as pd
import sys
import csv
from math import *
from abc import ABC, abstractmethod, abstractproperty
from scipy.special import factorial as spfactorial
import logging as log
import time   # for testing
import numpy as np
import scipy.optimize
import symengine
import math
import pandas as pd
import csv
from simulator import generator, common


parameters = {"IFRGSB": [0.1, 0.1], "GM": [0.01], "NB2": [0.01], "DW2": [0.994], "DW3": [0.1, 0.5], "S": [0.1, 0.1], "TL": [0.1, 0.1], "IFRSB": [0.1]}

def hazard_symbolic(model, i, args):
    if model == "GM":
        f = args[0]
        return f
    elif model == "DW3":
        f = 1 - symengine.exp(-args[0] * i**args[1])
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
            f = (1 - symengine.exp(-1/args[1])) / (1 + symengine.exp(- (i - args[0])/args[1]))
        except OverflowError:
            f = float('inf')
        return f

def RLL(model, x, num_hazard_params, kVec, covariates):
    kvec_sum = np.sum(kVec)
    fourthTerm = np.sum(np.log(spfactorial(kVec)))
    n = len(kVec)
    # the vector x contains [b, beta1, beta2, beta3, beta4] so 1: is just the betas
    betas = np.array(x[num_hazard_params:len(x)])
    hazard_params = x[0:num_hazard_params]
    prodlist = np.zeros(n)
    # store g calculations in an array, for easy retrieval
    glookups = np.zeros(n)
    for i in range(n):
        one_minus_hazard = (1 - common.hazard_numerical(model, i, hazard_params))
        try:
            glookups[i] = exp(np.dot(betas, covariates[:, i]))
        except OverflowError:
            return float('inf')
        # calculate the sum of all gxib from 0 to i, then raise (1 - b) to that sum
        exponent = np.sum(glookups[:i])
        sum1 = 1 - (one_minus_hazard ** glookups[i])
        prodlist[i] = (sum1 * (one_minus_hazard ** exponent))

    firstTerm = -kvec_sum  # Verified
    secondTerm = kvec_sum * np.log(kvec_sum/np.sum(prodlist))
    thirdTerm = np.dot(kVec, np.log(prodlist))  # Verified

    cv = (firstTerm + secondTerm + thirdTerm - fourthTerm)

    if np.isnan(cv):
        return float('inf')

    return cv
    
def LLF_sym(model, numSymbols, kVec, covariate_data):
    # x = b, b1, b2, b2 = symengine.symbols('b b1 b2 b3')
    n = len(kVec)
    num_hazard_params = numSymbols - len(covariate_data)
    x = symengine.symbols(f'x:{numSymbols}')
    second = []
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(num_hazard_params, numSymbols):
            TempTerm1 = TempTerm1 * symengine.exp(covariate_data[j - num_hazard_params][i] * x[j])
        sum1 = 1 - ((1 - (hazard_symbolic(model, i + 1, x[:num_hazard_params]))) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(num_hazard_params, numSymbols):
                TempTerm2 = TempTerm2 * symengine.exp(covariate_data[j - num_hazard_params][k] * x[j])
            sum2 = sum2 * ((1 - (hazard_symbolic(model, i + 1, x[:num_hazard_params])))**(TempTerm2))
        second.append(sum2)
        prodlist.append(sum1 * sum2)

    firstTerm = -sum(kVec)  # Verified
    secondTerm = sum(kVec) * symengine.log(sum(kVec) / sum(prodlist))
    logTerm = []  # Verified
    for i in range(n):
        logTerm.append(kVec[i] * symengine.log(prodlist[i]))
    thirdTerm = sum(logTerm)
    factTerm = []  # Verified
    for i in range(n):
        factTerm.append(symengine.log(math.factorial(kVec[i])))
    fourthTerm = sum(factTerm)

    f = firstTerm + secondTerm + thirdTerm - fourthTerm
    return f, x


def convertSym(x, bh, target):
    """Converts the symbolic function to a lambda function

    Args:

    Returns:

    """
    return symengine.lambdify(x, bh, backend='lambda')


def RLL_minimize(x, model, num_hazard_params, kVec, covariate_data):
    return -RLL(model, x, num_hazard_params, kVec, covariate_data)


def optimizeSolution(fd, B):
    # log.info("Solving for MLEs...")

    sol_object = scipy.optimize.root(fd, x0=B)
    solution = sol_object.x
    converged = sol_object.success
    # log.info("/t" + sol_object.message)

    return solution, converged


def modelFitting(betas, num_hazard_params, hazard, mle, kVec, covariate_data):
    omega = calcOmega(hazard, betas, kVec, covariate_data)
    # print(omega)
    # log.info("Calculated omega: %s", omega)

    mvf_array = MVF_all(mle, num_hazard_params, omega, hazard, kVec, covariate_data)
    # log.info("MVF values: %s", mvf_array)
    intensityList = intensityFit(mvf_array)
    # log.info("Intensity values: %s", intensityList)
    return omega, mvf_array


def calcOmega(h, betas, kVec, covariate_data):
    # can likely use fewer loops
    n = len(kVec)
    numCovariates = len(covariate_data)
    prodlist = []
    for i in range(n):
        sum1 = 1
        sum2 = 1
        TempTerm1 = 1
        for j in range(numCovariates):
            TempTerm1 = TempTerm1 * np.exp(covariate_data[j][i] * betas[j])
        sum1 = 1-((1 - h[i]) ** (TempTerm1))
        for k in range(i):
            TempTerm2 = 1
            for j in range(numCovariates):
                TempTerm2 = TempTerm2 * \
                    np.exp(covariate_data[j][k] * betas[j])
            sum2 = sum2*((1 - h[i])**(TempTerm2))
        prodlist.append(sum1*sum2)
    denominator = sum(prodlist)
    numerator = sum(kVec)

    return numerator / denominator


def MVF_all(mle, num_hazard_params, omega, hazard_array, kVec, covariate_data):
    mvf_array = np.array([MVF(mle, num_hazard_params, omega, hazard_array, dataPoints, covariate_data) for dataPoints in range(len(kVec))])
    return mvf_array


def MVF(x, num_hazard_params, omega, hazard_array, stop, cov_data):
    # gives array with dimensions numCovariates x n, just want n
    # switched x[i + 1] to x[i + numParameters] to account for
    # more than 1 model parameter
    # ***** can probably change to just betas
    exponent_all = np.array([cov_data[i][:stop + 1] * x[i + num_hazard_params] for i in range(len(cov_data))])

    # sum over numCovariates axis to get 1 x n array
    exponent_array = np.exp(np.sum(exponent_all, axis=0))

    h = hazard_array[:stop + 1]

    one_minus_hazard = (1 - h)
    one_minus_h_i = np.power(one_minus_hazard, exponent_array)
    one_minus_h_k = np.zeros(stop + 1)
    for i in range(stop + 1):
        k_term = np.array([one_minus_hazard[i] for k in range(i)])
        if len(cov_data) == 0:
            one_minus_h_k[i] = np.prod(
                np.array([one_minus_hazard[i]] * len(k_term)))
        else:
            exp_term = np.power((one_minus_hazard[i]), exponent_array[:][:len(k_term)])
            one_minus_h_k[i] = np.prod(exp_term)

    product_array = (1.0 - (one_minus_h_i)) * one_minus_h_k

    result = omega * np.sum(product_array)
    return result

def intensityFit(mvf_array):
    difference = [mvf_array[i+1]-mvf_array[i] for i in range(len(mvf_array) - 1)]
    return [mvf_array[0]] + difference

def runEstimation(model, num_hazard_params, kVec, covariateData):
    # need class of specific model being used, lambda function stored as class variable

    # ex. (max covariates = 3) for 3 covariates, zero_array should be length 0
    # for no covariates, zero_array should be length 3
    # numZeros = Model.maxCovariates - self.numCovariates
    # zero_array = np.zeros(numZeros)   # create empty array, size of num covariates

    # create new lambda function that calls lambda function for all covariates
    # for no covariates, concatenating array a with zero element array
    optimize_start = time.time()    # record time
    
    # Generate initial estimates
    parameterEstimates = list(parameters[model])
    betaEstimate = [0.01 for i in range(len(covariateData))]
    initial = np.array(parameterEstimates + betaEstimate)
    
    numSymbols = num_hazard_params + len(covariateData)

    # log.info("Initial estimates: %s", initial)
    # pass hazard rate function
    f, x = LLF_sym(model, numSymbols, kVec, covariateData)
	
    bh = np.array([symengine.diff(f, x[i]) for i in range(numSymbols)])

    fd = convertSym(x, bh, "numpy")

    solution_object = scipy.optimize.minimize(RLL_minimize, x0=initial, args=(model, num_hazard_params,kVec, covariateData,), method='Nelder-Mead')
    mle_array, converged = optimizeSolution(fd, solution_object.x)
    # print(mle_array)
    optimize_stop = time.time()
    runtime = optimize_stop - optimize_start
    # print(runtime)

    modelParameters = mle_array[:num_hazard_params]
    # print(modelParameters)
    betas = mle_array[num_hazard_params:]
    # log.info("model parameters =", modelParameters)
    # log.info("betas =", betas)
    hazard = np.array([common.hazard_numerical(model, i + 1, modelParameters) for i in range(len(kVec))])
    hazard_array = hazard    # for MVF prediction, don't want to calculate again
    omega, mvf_array = modelFitting(betas, num_hazard_params, hazard, mle_array, kVec, covariateData)
    return omega, mvf_array, converged, mle_array


def PSSE(covariates, omega, hazard_params, num_covariates, betas, kVec, model):
    full_length = len(kVec)
    kVecNew = kVec[:full_length-2]
    mvf = 0
    accumulator = 0
    truncated_length = len(kVecNew)
    truncated_sum = sum(kVecNew)
    PSSE = 0
    for i in range(truncated_length, full_length):
        mvf = float(omega*generator.p(model, hazard_params,
                    i, covariates, num_covariates, betas))
        accumulator += mvf
        PSSE += (mvf-kVec[i-1])**2
#        print(f"[+] Predicted FC at interval {i+1} is: {mvf}")
#        print(f"[+] Actual FC at interval {i+1} is: {kVec[i-1]}\n")
#    print(
#        f"\n-----------Predictive cumulative FC is {truncated_sum+accumulator}---------------")
#    print(
#        f"-------------Actual cumulative FC is {sum(kVec)}------------------------------")
#    print(
#        f"-------------PSSE: {PSSE}-------------------------------------------------\n")
    if PSSE > 255:
        return 255
    else:
        return PSSE
