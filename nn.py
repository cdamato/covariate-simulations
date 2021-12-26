
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import torch
import torchinfo
from torchinfo import summary
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt

import sys
import os
from simulator import generator, Facilitator

num_hazards = 3
num_intervals = 25
num_covariates = 2

def sortHazard(values): 
    '''Uses a dictionary to map results vector psse to the corresponding hazard function.
Also is utilized to map output of NN to the corresponding hazard function.
'''
    values = np.array(values.data.numpy()).transpose()  
    hazards = {
        'GM': values.item(0),
        'DW3': values.item(1),
        'DW2': values.item(2),
        # 'NB2': 48,
        # 'S': 41,
        # 'IFRSB' : 20,
        # 'IFRSGB' : 10,
    }
    return sorted(hazards.items(), key=lambda x:x[1]) #sorts the output of the dictionary in ascending order by values

# def sortoutput(output):
'''Dont worry about this for now, 
the functionality of this is taken care of within the sortHazard function, 
this was just my initial thought process'''
#     psse_array = [None] * num_hazards
#     array = np.array(output.data.numpy())
#     for num in range(0, num_hazards):          
#        psse_array[num] = array.item(num)
#     updated_hazard = sortHazard(psse_array)
#     return updated_hazard


# Definition of the training network
class ANN(nn.Module):
    def __init__(self, input_dim=num_intervals, output_dim=num_hazards):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim+5)
        self.fc2 = nn.Linear(input_dim+5, input_dim+5)
        self.fc3 = nn.Linear(input_dim+5, input_dim)
        self.output_layer = nn.Linear(input_dim, num_hazards)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.output_layer(x)
        return nn.Sigmoid()(x)

# Generates a training dataset, split into inputs and results.
# Input will be a matrix of size <num_covariates> by <num_intervals>, containing a generated dataset.
# Results will be a matrix 1 by <num_hazards>, containing ideal evaluated outputs.
# NOTE: This function can be generalized by passing in an evaluation function to build results


def gen_training_detaset(epoch):
    model_id = random.randint(0, num_hazards - 1)  # Pick a model
    models = ["GM", "DW3", "DW2", "NB2", "S", "IFRSB", "IFRGSB"]
    results = np.array([[0.0]] * num_hazards).transpose() #Intended output vector, which the loss function is measured  against
    results = np.zeros(shape=(num_covariates+1,num_hazards)) #Intended output vector
    training_input = generator.simulate_dataset(models[model_id], num_intervals, num_covariates)
    # print(f"\nModel is {models[model_id]}")
    # plt.title(models[model_id])
    # plt.plot(training_input[0], color="red")
    # print(f"At epoch {epoch}, For {models[model_id]}, kvec is {training_input[0]}\n")
    # plt.savefig(f"DatasetPlots/{models[model_id]}Epoch{epoch}.png")
    # plt.close()
    for covariates in range (num_covariates+1):
     for index in range(num_hazards):
            results[covariates, index] = Facilitator.MaximumLiklihoodEstimator(models[index], training_input[:covariates+1])
            #print(f"Result vector for {models[index]} is {results[0, index]}")
    training_input = torch.from_numpy(training_input)
    training_output = torch.from_numpy(results)
    train = torch.utils.data.TensorDataset(training_input, training_output)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=1, shuffle=True)
    return train_loader

# Normalizes a tensor so that the sum of all elements is 100%


def normalize_tensor_to_100(tensor):
    max_out = max(1e-64, sum(tensor[0].tolist()))
    return [i * (1 / max_out) for i in output][0]


model = ANN()
summary(model)
#loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
epochs = 100

# I am REALLY not sure what these are for
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

 # prepare model for training
# for epoch in range(epochs):
#     # are all these necessary?
#     trainloss = 0.0
#     valloss = 0.0
#     correct = 0
#     total = 0
#     train_loader = gen_training_detaset(epoch)
#     #model.train()
#     # there must be SOME way to clean this up...
#     for data, target in train_loader:
#         data = Variable(data).float()
#         target = Variable(target).type(torch.FloatTensor)
#         optimizer.zero_grad()
#         output = model(data)
#         print("Output vector is", output)
#         predicted = (torch.round(output.data[0]))
#         total += len(target)
#         correct += (predicted == target).sum()

#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#         trainloss += loss.item()*data.size(0)

# 	# What is any of this for either?
#     trainloss = trainloss/len(train_loader.dataset)
#     accuracy =  correct / float(total)
#     train_acc_list.append(accuracy)
#     train_loss_list.append(trainloss)
#     print('Epoch: {} \tTraining Loss: {:.4f}\t'.format(
#         epoch+1,
#         trainloss))
#     epoch_list.append(epoch + 1)
#model.train()
min_valid_loss = np.inf
loss_array = [None] * epochs
val_array = [None] * epochs 
for e in range(epochs):
    train_loss = 0.0
    trainloader = gen_training_detaset(e)
    validloader = gen_training_detaset(e)
    for data, target in trainloader:
        #sorted_results = sortoutput(target)
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        output = model(data)
        #sorted_output = sortoutput(output) #uses dictionary to sort the models output/ for training
        #print(f"[+] Sorted output for results: {sorted_results}\n[+] Sorted output for model prediction {sorted_output}\n")
        #print("Training Output vector is", output)
        # Find the Loss
        loss = loss_fn(output, target)
        loss_array[e] = loss.item()
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    cov_count = 0
    for data, target in validloader:
        sorted_results = sortHazard(target)
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # Forward Pass
        output = model(data)
        # uses dictionary to sort the models output/ for validation. Same process can be done for training but that is not really interesting.
        sorted_output = sortHazard(output)
        print(f"!! {cov_count} covariates!!\n[+] Sorted output for results: {sorted_results}\n[+] Sorted output for model prediction {sorted_output}\n")
        cov_count += 1
        #print("Validation output vector is", output)
        # Find the Loss
        loss = loss_fn(output, target)
        val_array[e] = loss.item()
        # Calculate Loss
        valid_loss += loss.item()
    print(f'----Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}------\n')

    if min_valid_loss > valid_loss:
        print(f'------Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model------\n')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax.set_title('Training Loss vs Epoch')
ax.plot(loss_array, color="red")
#ax.savefig(f"TrainingLossvsEpoch.png")    #Can be consolidated by using a function, maybe called graphResults(loss_array, val_array).
ax2.set_title('Validation Loss vs Epoch')
ax2.plot(val_array, color="red")
#ax2.savefig(f"ValidationLossvsEpoch.png")
plt.savefig(f"ValandTrain.png")
plt.show()
