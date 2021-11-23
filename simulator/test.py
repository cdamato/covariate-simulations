import csv, RLLs, convergence, pandas as pd, numpy as np, time

parameters = {"IFRGSB": [0.1, 0.1], "GM": [0.01], "NB2": [0.01], "DW2": [0.994], "DW3": [0.1, 0.5], "S": [0.1, 0.1], "TL": [0.1, 0.1], "IFRSB": [0.1]}
num_covariates = 10
num_simulated_sets = 1000

# Do not modify!
def test_RLL():
    for hazard_name in common.models:
        for model in common.models:
            for numCov in range(1, num_covariates+1):
                runtimes = []
                for run in range(1, num_simulated_sets+1):
                    input_file = f"ds1.csv"
                    dataset = pd.read_csv(input_file).transpose().values
                    dataset = dataset[1:]
                    covariates = dataset[1:numCov]
                    kVec = dataset[0]
                    
                    metricNames = csv.reader(open(input_file, newline=''))
                    metricNames = next(metricNames)[2:]
                    data = pd.read_csv(input_file)
                    t = data["T"].values     # failure times
                    kVec = data["FC"].values     # number of failures
                    totalFailures = sum(kVec)
                    n = len(kVec)
                    covariates = np.array([data[name].values for name in metricNames])
                    numCovariates = len(covariates)
                    num_hazard_params = len(parameters[model])
                    runtime, converged, mle_array = convergence.runEstimation(model, num_hazard_params, kVec, covariates)
                    
  
                    if converged:
                        objective_start = time.time()
                        LL = convergence.RLL(model, mle_array, num_hazard_params, kVec, covariates)
                        objective_stop = time.time()
                        time2 = objective_stop - objective_start
 
                        print(f"RLL is {LL}")


func()
