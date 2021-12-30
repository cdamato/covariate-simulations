import csv
from numpy.core.numeric import Inf, Infinity
import pandas as pd
import numpy as np
import time
from simulator import generator, convergence
import matplotlib
import matplotlib.pyplot as plt

parameters = {"IFRGSB": [0.1, 0.1], "GM": [0.01], "NB2": [0.01], "DW2": [
    0.994], "DW3": [0.1, 0.5], "S": [0.1, 0.1],  "IFRSB": [0.1]}
hazard_names = ["IFRGSB", "GM", "NB2", "DW2", "DW3", "S", "IFRSB"]


def MaximumLiklihoodEstimator(model, input_dataset, iteration ,start ,end):
    # model = "GM"
    # base_directory = "TEST"
    # num_intervals = 25
    # Covariates = 3
    # dataset_name = generator.print_dataset(generator.simulate_dataset(
    #     model, num_intervals, Covariates), num_intervals, base_directory, model, Covariates)
    # metricNames = csv.reader(open(dataset_name, newline=''))
    # metricNames = next(metricNames)[2:]
    # data = pd.read_csv(dataset_name)
    # kVec = data["FC"].values     # number of failures
    # covariates = np.array(
    #     [data[name].values for name in metricNames])
    # numCovariates = len(covariates)
    # num_hazard_params = len(parameters[model])
    '''IF (iteration % numcovariates) = 2, then drop the covariates bewteen start and end'''
    if iteration != 0:
        test_beg = start[iteration]
        test_end = end[iteration]
        covariates = input_dataset[test_beg:(test_end)+1, :]
        if iteration == 5:
            covariates = np.zeros(shape=(2, 25))
            covariates[0,:] = input_dataset[test_beg, :]
            covariates[1,:] = input_dataset[(test_end), :]
            
    else:
        covariates = input_dataset[1:]
    kVec = input_dataset[0,:]
    num_hazard_params = len(parameters[model])
    Omega, mvf_array, converged, mle_array = convergence.runEstimation(
        model, num_hazard_params, kVec, covariates)
    Hazard_params = mle_array[0:num_hazard_params]
    betas = mle_array[num_hazard_params:]
    if converged:
        PSSE = convergence.PSSE(covariates, Omega, Hazard_params, len(covariates), betas, kVec, model)
        return PSSE
    else:
        return 2**32-1

    # if converged:
    #     print(
    #         f"Data:{dataset_name} | Model:{model} | {numCovariates} Covariate(s)\n")
    #     print(
    #         f" [+] Omega: {Omega}\n [+] Hazard Parameters: {Hazard_params}\n [+] MLE Array: {betas}\n ")
    #     convergence.PSSE(covariates, Omega, Hazard_params, numCovariates, betas, kVec)
    # else:
    #     print(f'-----[!] WARNING {dataset_name} | Model:{model} | {numCovariates} Covariates DID NOT CONVERGE!-----\n\n')
        


