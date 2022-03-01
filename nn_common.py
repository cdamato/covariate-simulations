import random
import numpy as np
from sklearn import model_selection
from simulator import generator, common, fitting
import torch

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
def gen_training_detaset(epoch, num_hazards, num_covariates, num_intervals, validation):
    model_id = random.randint(0, num_hazards - 1)  # Pick a model
    dataset = generator.simulate_dataset(common.models[model_id], num_intervals, num_covariates)
    results = np.zeros(shape=(num_hazards, 1))
    # Calculate PSSE of all covariate/hazard combos using numerical methods, and store it in `results`
    # results = np.zeros(shape=(2**num_covariates,num_hazards))
    # subset_list = np.zeros(shape=(2 ** num_covariates, num_intervals * (num_covariates + 1)))
    # for combination in range(2**num_covariates):
    #     subset = covariates_subset(dataset, num_covariates, combination) 
    #     subset_list[combination] = (np.resize(subset, (num_covariates+1,num_intervals)).flatten()) / np.amax(subset) # what is happening here?
    for model_index, model in enumerate(common.models[0:num_hazards]):
        results[model_index, 0] = fitting.MaximumLiklihoodEstimator(model, dataset) # why not PSSE?
    #datasetPlotter(common.models[model_id], dataset[0], epoch, validation)
    # Why are we overwriting the dataset? and what is it being overwritten with? what
    training_input = np.reshape(dataset, newshape=(1,(1+num_covariates)*num_intervals)) 
    training_input = training_input / np.amax(training_input)
    training_output = np.reshape(results, newshape=(1,num_hazards))
    training_input = torch.from_numpy(training_input)
    training_output = torch.from_numpy(training_output)
    train = torch.utils.data.TensorDataset(training_input, training_output)
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
    return train_loader


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

	
def datasetPlotter(model, FC, epoch, validation):
    plt.title(model)
    plt.plot(FC, color="red")
    if validation == True:
        plt.savefig(f"DatasetPlots/VAL-{model}Epoch{epoch}.png")
    else:
        plt.savefig(f"DatasetPlots/{model}Epoch{epoch}.png")
        plt.close()
    return
