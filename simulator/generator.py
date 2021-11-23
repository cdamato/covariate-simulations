import scipy.stats
import numpy as np
from csv import writer
import random
import sys, os
import random, math
import pandas as pd
import numpy as np
import common

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

def g(x, n, betas, interval): # Equation 15
	g = 0
	for i in range(0, n):
		g += betas[i] * x[i][interval]
	g = math.exp(g)
	return g

def p(model, params, interval, x, n, beta): # Equation 19
	pixi = 1 - pow(1 - common.hazard_numerical(model, interval + 1, params), g(x, n, beta, interval))
	for k in range(0, interval):
		pixi *= pow(1 - common.hazard_numerical(model, k + 1, params), g(x, n, beta, k))
	return pixi
	
def generate_FC(model_name, x, covNum, num_intervals, beta, omega, hazard_params):
	n = len(beta)
	failures = np.zeros(num_intervals)
	cumulative = 0
	for j in range(num_intervals):	
		prob = p(model_name, hazard_params, j, x, n, beta)
		failures[j] = int(poisson_variate(omega * prob))
		cumulative += prob
	return failures


def generate_hazardparams(hazard_name):
	param_ranges = {
		"GM":  [random.uniform(0.0147, 0.059)],
		"NB2": [random.uniform(0.079, 0.123)],
		"DW2": [random.uniform(0.995, 0.997)],
		"DW3": [random.uniform(0.005, 0.015), random.uniform(0.005, 0.015)],
		"S":   [random.uniform(0.018, 0.103), random.uniform(0.762, 0.911)],
		"TL":  [random.uniform(6.143, 19.418), random.uniform(6.062, 36.985)],
		"IFRSB": [random.uniform(0.993, 0.995)],
		"IFRGSB": [random.uniform(0.972, 0.999), random.uniform(0, 0.003)]
	}
	return param_ranges[hazard_name]


def generate_betas(num_covariates):
	betas = []
	for covariate in range(num_covariates):
		betas.append(random.uniform(0.01, 0.05))
	return betas;

def generate_omega(hazard_name):
	omega_ranges = {
		"GM":  random.uniform(54.217, 93.231),
		"NB2": random.uniform(71.845, 87.941),
		"DW2": random.uniform(93.753, 103.909),
		"DW3": random.uniform(67.015, 93.754),
		"S":   random.uniform(64.709, 89.747),
		"TL":  random.uniform(55.246, 96.743),
		"IFRSB": random.uniform(105.828, 137.003),
		"IFRGSB": random.uniform(64.483, 100.000),
	}
	return omega_ranges[hazard_name]


def simulate_dataset(hazard, num_intervals, num_covariates):
    cov_dataset = np.zeros((num_covariates + 1, num_intervals))

    for covariate in range(num_covariates):
        for interval in range(num_intervals):
            cov_dataset[covariate, interval] = scipy.stats.expon.rvs(random.randint(2, 7))

    cov_dataset[0,:] = generate_FC(hazard, cov_dataset[1:,:], num_covariates, num_intervals, \
        generate_betas(num_covariates), generate_omega(hazard), generate_hazardparams(hazard))

    return cov_dataset;


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

batch_simulator("dir", 10, 10, 10)
