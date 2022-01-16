import numpy as np
from scipy.stats.stats import gmean
import torch
from torchinfo import summary
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
from simulator import generator, Facilitator, common
import itertools

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


def covariates_subset(dataset, num_covariates, combo_index):
	# Given a combination number, determine which covarites are active.
	# This uses binary arithmetic to find all true bits
	# Counting from 0 to 2^n, where n is the number of covariates, will give all combination indices
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
	
def datasetPlotter(model, FC, epoch, validation):
    plt.title(model)
    plt.plot(FC, color="red")
    if validation == True:
     plt.savefig(f"DatasetPlots/VAL-{model}Epoch{epoch}.png")
    else:
     plt.savefig(f"DatasetPlots/{model}Epoch{epoch}.png")
    plt.close()
	
# Definition of the training network
class ANN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim*2)
        self.fc2 = torch.nn.Linear(input_dim*2, input_dim*4)
        self.fc3 = torch.nn.Linear(input_dim*4, input_dim*5)
        self.fc4 = torch.nn.Linear(input_dim*5, input_dim*6)
        self.fc5 = torch.nn.Linear(input_dim*6, input_dim*5)
        self.fc6 = torch.nn.Linear(input_dim*5, input_dim*4)
        self.fc7 = torch.nn.Linear(input_dim*4, input_dim*3)
        self.fc8 = torch.nn.Linear(input_dim*3, input_dim)
        self.output_layer = torch.nn.Linear(input_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.15)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = self.output_layer(x)
        # print(x)
        # print(torch.relu(x))
        # print(torch.clamp(x, min= 0, max =255))
        return x

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
            results[combination, model_index] = Facilitator.MaximumLiklihoodEstimator(model, subset)
    #print(subset_list)
    #training_input.resize(2**num_covariates, num_intervals, refcheck=False)
    datasetPlotter(common.models[model_id], training_input[0], epoch, validation)
    training_input = torch.from_numpy(subset_list)
    training_output = torch.from_numpy(results)
    train = torch.utils.data.TensorDataset(training_input, training_output)
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
    return train_loader

def train_model(num_hazards, num_covariates, num_intervals, learning_rate, weight_decay, output_directory):
    nn = ANN(num_intervals*(num_covariates+1), num_hazards)
    #summary(nn)
    if torch.cuda.is_available():
        nn.cuda() 
    loss_fn = torch.nn.L1Loss()
    #loss_fn = nn.MSELoss()
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs = 10


    accumulators = {}
    for index, m in enumerate(common.models[:num_hazards]):
        accumulators[m] = []
    min_valid_loss = np.inf
    loss_array = [None] * int(epochs/25)
    val_array = [None] * int(epochs/25) 
    for e in range(epochs):
        train_loss = 0.0
        cov_count = 0
        trainloader = gen_training_detaset(e , num_hazards, num_covariates, num_intervals, False)
        for data, target in trainloader:
            sorted_results = sortHazard(target)
            data = Variable(data).float()
            target = Variable(target).type(torch.FloatTensor)
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            output = nn(data)
            sorted_output = sortHazard(output)
            # Find the Loss
            loss = loss_fn(output, target)
            if e % 25 == 0:
                loss_array.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
            print(f"!! covariate vector: {bin(cov_count)} !!\n[+] Sorted output for results: {sorted_results}\n[+] Sorted output for model prediction {sorted_output}\n")
            cov_count+=1
            print(f'- ---Epoch {e+1} \t\t Training Loss: {loss}\n')
    print("\n-------------------- VALIDATION LOOP -----------------------\n")
    valid_loss = 0.0
    cov_count = 0
    with torch.no_grad():
        nn.eval()
        for i in range(epochs):
            validloader = gen_training_detaset(
                i, num_hazards, num_covariates, num_intervals, True)
            cov_count = 0
            for data, target in validloader:
                sorted_results = sortHazard(target)
                data = Variable(data).float()
                target = Variable(target).type(torch.FloatTensor)
                # Transfer Data to GPU if available
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Forward Pass
                output = nn(data)
                # uses dictionary to sort the models output/ for validation. Same process can be done for training but that is not really interesting.
                sorted_output = sortHazard(output)
                output_accumulator = [item[0] for item in sorted_output]
                for count in range(len(sorted_results)):
                    model_name = sorted_results[count][0]
                    if output_accumulator[count] == model_name:
                        accumulators[model_name].append(count)
                
                print(f"!! covariate vector: {bin(cov_count)} !!\n[+] Sorted output for results: {sorted_results}\n[+] Sorted output for model prediction {sorted_output}\n")
                cov_count += 1
                # Find the Loss
                loss = loss_fn(output, target)
                if i % 25 == 0:
                    val_array.append(loss.item())
                # Calculate Loss
                valid_loss += loss.item()
                print(f'----Epoch {i+1} Validation Loss: {loss}\n')

            if min_valid_loss > valid_loss:
                print(f'------Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model------\n')
                min_valid_loss = valid_loss

                # Saving State Dict
                torch.save(nn.state_dict(), 'saved_model.pth')



        
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.set_title('Training Loss vs Epoch')
    ax.plot(loss_array, color="red")
    #ax.savefig(f"TrainingLossvsEpoch.png")    #Can be consolidated by using a function, maybe called graphResults(loss_array, val_array).
    ax2.set_title('Validation Loss vs Epoch')
    ax2.plot(val_array, color="red")
    #ax2.savefig(f"ValidationLossvsEpoch.png")
    total_chances = epochs * 2**num_covariates
    plt.savefig(f"{output_directory}/ValandTrain.png")


    plt.close()
    for model in common.models[:num_hazards]:
        plt.title(f"{model} - Amount and Locations of Accurate Prediction")
        plt.hist(accumulators[model], edgecolor='black')
        plt.plot()
        plt.savefig(f"{output_directory}/{model} - Amount and Locations of Accurate Prediction")
        plt.close()
    
    
    for model in common.models[:num_hazards]:
        print(f"{model} accuracy: {len(accumulators[model])/total_chances}\n")
        



#(num_hazards , num_covariates , num_intervals , learning_rate , weight_decay , output_directory)
train_model(3, 2, 25, 0.001, 1e-6,"sim1")
