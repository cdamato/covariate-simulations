import scipy.stats
import numpy as np
from csv import writer
import random
import sys, os
import random, math
import pandas as pd
import numpy as np
from simulator import common

def poisson_variate(lam): # algorithm to find pseudorandom variate of the Poisson distribution
	x = 0
	p = math.exp(-lam)
	s = p
	u = random.uniform(0, 1)
	
	while u > s:
		x += 1
		p *= lam / x
		s += p
	return x

def generate_FC(model_name, x, covNum, num_intervals, beta, omega, hazard_params):
	n = len(beta)
	failures = np.zeros(num_intervals)
	cumulative = 0
	for j in range(num_intervals):	
		prob = common.p(model_name, hazard_params, j, x, n, beta)
		failures[j] = int(poisson_variate(omega * prob))
		cumulative += prob
	return failures

# Most models have an actual parameter range between 0 and 1.
# However, they act strange near the edge cases, so clamping the values produces more identifiable results.
# As of now, TL, IFRSB, and IFRSGB have not had tailored parameter estimates.
def generate_hazardparams(hazard_name):
	param_ranges = {
		"GM":  [random.uniform(0.025, 0.25)],
		"NB2": [random.uniform(0.025, 0.2)],
		"DW2": [random.uniform(0.95, 0.99)],
		"DW3": [random.uniform(0.05, 0.75), random.uniform(-1, 1)],
		"S":   [random.uniform(0.05, 0.5), random.uniform(0.75, 0.9)],
		#"TL":  [random.uniform(6.143, 19.418), random.uniform(6.062, 36.985)],
		"IFRSB": [random.uniform(0.2, 0.8)],
		"IFRGSB": [random.uniform(0.972, 0.999), random.uniform(0, 0.003)]
	}
	return param_ranges[hazard_name]

# What are some good values for betas?
def generate_betas(num_covariates):
	betas = []
	for covariate in range(num_covariates):
		betas.append(random.uniform(0.01, 0.05))
	return betas;

# I believe omega is independent of model, so I removed the dictionary.
# If this is false, it can be re-acquired from git.
def generate_omega(hazard_name):
	return random.uniform(25, 125)

# Input will be a matrix of size <num_covariates> by <num_intervals>, containing a generated dataset.
def simulate_dataset(hazard, num_intervals, num_covariates):
	cov_dataset = np.zeros((num_covariates + 1, num_intervals))
	for covariate in range(num_covariates):
		for interval in range(num_intervals):
			cov_dataset[covariate+1, interval] = scipy.stats.expon.rvs(random.randint(2, 7))
	cov_dataset[0,:] = generate_FC(hazard, cov_dataset[1:,:], num_covariates, num_intervals,generate_betas(num_covariates), generate_omega(hazard), generate_hazardparams(hazard))
	return cov_dataset


def print_dataset(cov_dataset, num_intervals, base_directory, model, num_covariates):
	try:
		os.mkdir(f"{base_directory}/{model}", 0o777)
	except:
		pass

	output_filename = f"{base_directory}/{model}/{num_covariates}cov.csv"

	with open(output_filename, 'w') as myfile:
		wr = writer(myfile)
		wr.writerow(['T', 'FC'] + [f'x{i+1}' for i in range(num_covariates)])
		wr.writerows(zip(range(1, num_intervals + 1), *cov_dataset))
		
def batch_simulator(base_directory, num_intervals, max_num_covariates, num_datasets):
	try:
		os.mkdir(base_directory)
	except:
		print("failed creating base directory? might still work")

	for i in range(num_datasets):
		sim_dir = f"{base_directory}/sim{str(i + 1)}"
		try:
			os.mkdir(sim_dir)
		except:
			print("failed creating simulation directory")
			pass

		for model in common.models:
			for num_covariates in range(max_num_covariates + 1):
				cov_dataset = simulate_dataset(model, num_intervals, num_covariates)
				print_dataset(cov_dataset, num_intervals, sim_dir, model, num_covariates)


#batch_simulator("datasets", 25, 5, 25)
