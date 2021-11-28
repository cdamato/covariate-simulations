import csv
import convergence
import pandas as pd
import numpy as np
import time
import generator

parameters = {"IFRGSB": [0.1, 0.1], "GM": [0.01], "NB2": [0.01], "DW2": [
    0.994], "DW3": [0.1, 0.5], "S": [0.1, 0.1], "TL": [0.1, 0.1], "IFRSB": [0.1]}
hazard_names = ["IFRGSB", "GM", "NB2", "DW2", "DW3", "S", "TL", "IFRSB"]


def MaximumLiklihoodEstimator():
    model = "GM"
    base_directory = "TEST"
    num_intervals = 25
    Covariates = 3
    dataset_name = generator.print_dataset(generator.simulate_dataset(
        model, num_intervals, Covariates), num_intervals, base_directory, model, Covariates)
    metricNames = csv.reader(open(dataset_name, newline=''))
    metricNames = next(metricNames)[2:]
    data = pd.read_csv(dataset_name)
    kVec = data["FC"].values     # number of failures
    covariates = np.array(
        [data[name].values for name in metricNames])
    numCovariates = len(covariates)
    num_hazard_params = len(parameters[model])
    Omega, mvf_array, converged, mle_array = convergence.runEstimation(
        model, num_hazard_params, kVec, covariates)
    Hazard_params = mle_array[0:num_hazard_params]
    betas = mle_array[num_hazard_params:]
    

    if converged:
        print(
            f"Data:{dataset_name} | Model:{model} | {numCovariates} Covariate(s)\n")
        print(
            f" [+] Omega: {Omega}\n [+] Hazard Parameters: {Hazard_params}\n [+] MLE Array: {betas}\n ")
        convergence.MeanValueFunction(covariates, Omega, Hazard_params, numCovariates, betas, kVec)
    else:
        print(f'-----[!] WARNING {dataset_name} | Model:{model} | {numCovariates} Covariates DID NOT CONVERGE!-----\n\n')
        
for i in range(1,10):
    MaximumLiklihoodEstimator()

