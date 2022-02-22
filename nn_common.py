import random
import numpy as np
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
    results = np.zeros(shape=(2**num_covariates,num_hazards))
    training_input = generator.simulate_dataset(common.models[model_id], num_intervals, num_covariates)
    subset_list = np.zeros(shape=(2**num_covariates, num_intervals*(num_covariates +1)))
    for index, combination in enumerate(range(2**num_covariates),0):
        subset = covariates_subset(training_input, num_covariates, combination) 
        subset_list[index] = (np.resize(subset, (num_covariates+1,num_intervals)).flatten()) /np.amax(subset)
        for model_index, model in enumerate(common.models[0:num_hazards]):
            results[combination, model_index] = fitting.MaximumLiklihoodEstimator(model, subset)
    #print(subset_list)
    #training_input.resize(2**num_covariates, num_intervals, refcheck=False)
    datasetPlotter(common.models[model_id], training_input[0], epoch, validation)
    training_input = torch.from_numpy(subset_list)
    training_output = torch.from_numpy(results)
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
  #  plt.title(model)
  #  plt.plot(FC, color="red")
  #  if validation == True:
  #   plt.savefig(f"DatasetPlots/VAL-{model}Epoch{epoch}.png")
  #  else:
  #   plt.savefig(f"DatasetPlots/{model}Epoch{epoch}.png")
  #  plt.close()
  return
