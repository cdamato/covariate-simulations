import random
import numpy as np
from sklearn import model_selection
from simulator import generator, common, fitting
import torch
import matplotlib.pyplot as plt

# This treats combo_index as a bitstring, where set bits indicate the covariate with that index is active
# Counting from 0 to 2^n, where n is the number of covariates, will give all combinations
def covariates_subset(dataset, num_covariates, combo_index):
    active = []
    counter = 0
    index = combo_index
    while (index != 0):
        if (index & 1 == 1): 
            active.append(counter)
        index >>= 1
        counter += 1
    results = np.zeros(shape=(num_covariates + 1, len(dataset[0])))
    results[0] = dataset[0] # copy failure count
    for index, value in enumerate(active):
        results[index + 1] = dataset[value + 1]
    return results
    
# Generates a training dataset, split into inputs and results.
# Input will be a matrix of size <num_covariates> by <num_intervals>, containing a generated dataset.
# Results will be a matrix 1 by <num_hazards>, containing ideal evaluated outputs.
# NOTE: This function can be generalized by passing in an evaluation function to build results
def gen_training_detaset(epoch, num_hazards, num_covariates, num_intervals, validation, batch_size):
    results = np.zeros((batch_size, num_hazards))
    models = np.zeros(batch_size)
    training_input = np.zeros((batch_size, (1 + num_covariates) * num_intervals))
    #training_output = np.zeros(batch_size, 1, num_hazards)

    for batch_num in range(batch_size):
        model_id = random.randint(0, num_hazards-1)
        models[batch_num] = model_id
        dataset = generator.simulate_dataset(common.models[model_id], num_intervals, num_covariates)
        # Calculate PSSE of all covariate/hazard combos using numerical methods, and store it in `results`
        # results = np.zeros(shape=(2**num_covariates,num_hazards))
        # subset_list = np.zeros(shape=(2 ** num_covariates, num_intervals * (num_covariates + 1)))
        # for combination in range(2**num_covariates):
        #     subset = covariates_subset(dataset, num_covariates, combination)
        #     subset_list[combination] = (np.resize(subset, (num_covariates+1,num_intervals)).flatten()) / np.amax(subset) # what is happening here?
        for model_index, model in enumerate(common.models):
            results[batch_num, model_index] = fitting.MaximumLiklihoodEstimator(model, dataset) # why not PSSE?
        datasetPlotter(common.models[model_id], dataset[0], epoch, validation, batch_num)
        training_input[batch_num] = dataset.flatten()
        # Why are we overwriting the dataset? and what is it being overwritten with? what

    # training_input = training_input / np.amax(training_input)
    #training_output = np.reshape(results, newshape=(batch_size,num_hazards))
    training_input = torch.from_numpy(training_input)
    training_output = torch.from_numpy(results)
    #apply linear normalization to the target
    training_output /= 127
    train = torch.utils.data.TensorDataset(training_input, training_output)
    train_loader = torch.utils.data.DataLoader(train, shuffle=False)

    return models, train_loader

def sortHazard(values): 
    '''Uses a dictionary to map results vector psse to the corresponding hazard function.
Also is utilized to map output of NN to the corresponding hazard function.''' 
    if torch.cuda.is_available():
        values = np.array(values.data.cpu().numpy()).transpose()
    else:
        values = np.array(values.data.numpy()).transpose()
    hazard_mapping = {}
    for index, model in enumerate(values):
        hazard_mapping[common.models[index]] = values.item(index)
    return sorted(hazard_mapping.items(), key=lambda x:x[1]) #sorts the output of the dictionary in ascending order by values

	
def datasetPlotter(model, FC, epoch, validation, batch_num):
    plt.title(model)
    plt.plot(FC, color="red")
    fname = ""
    if validation:
        fname = f"DatasetPlots/VAL-{model}-{batch_num}Epoch{epoch}.png"
    else:
        fname = f"DatasetPlots/{model}-{batch_num}Epoch{epoch}.png"
    plt.savefig(fname)
    plt.close()
    return
