#!/usr/bin/python3
from simulator import fitting, common
import nn_common

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable 
import matplotlib.pyplot as plt
import sys, os, itertools
import plotly.express as px 
	
# Definition of the training network
class ANN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        loss_fn = None
        optimizer = None
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim *2)
        self.fc2 = torch.nn.Linear(input_dim *2, input_dim)
        self.fc3 = torch.nn.Linear(input_dim , input_dim*2)
        #self.fc4 = torch.nn.Linear(input_dim * 2, input_dim)
        self.output_layer = torch.nn.Linear(input_dim*2, output_dim)
        self.dropout = torch.nn.Dropout(0.55)

    def swish(self, x):
      return x * torch.sigmoid(x)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x) 
        #x = self.swish(self.fc4(x))
        x = self.output_layer(x)
        return x

    def run_epoch(self, data, target):
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = self(data)
        return nn_common.sortHazard(output), self.loss_fn(output, target)
        


def epoch(nn, train_loader, location,  validation):
    train_loss = 0.0
    cov_count = 0
    column = 0
    #want to run through multiple different graphs every single epoch
    for data, target in train_loader:
        # Clear the gradients
        numerical_results = nn_common.sortHazard(target)
        nn.optimizer.zero_grad()
        model_results, loss = nn.run_epoch(data, target)

        if validation:
            # uses dictionary to sort the models output/ for validation. Same process can be done for training but that is not really interesting.
            calculated_best_hazard = numerical_results[0][0] 
            predicted_best_hazard = model_results[0][0]
            for nn_model_index, model in enumerate(common.models):
                if model == predicted_best_hazard:
                    column = nn_model_index
                    break

            for i, hazard in enumerate(model_results):
                if hazard[0] == numerical_results[0][0]: # tuples
                  location[i] += 1
        else:
            # Calculate gradients
            loss.backward()
            # Update Weights
            nn.optimizer.step()
        # Calculate Loss
        train_loss += loss.item()
        print(f"[+] covariate vector: {bin(cov_count)} [+]")
        print(f"[+] Sorted output for results: {numerical_results} [+]")
        print(f"[+] Sorted output for model prediction {model_results} [+] \n")
        cov_count += 1
    return column, train_loss  
          
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def train_model(num_hazards, num_covariates, num_intervals, learning_rate, weight_decay, output_directory):
    epochs = 100
    nn = ANN(num_intervals*(num_covariates+1),num_hazards)
    nn.loss_fn = torch.nn.CrossEntropyLoss()
    nn.optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if torch.cuda.is_available():
        nn.cuda() 
  
    loss_array = [0] * int(epochs)
    for e in range(epochs):
        model_id, train_loader = nn_common.gen_training_detaset(e, num_hazards, num_covariates, num_intervals, False, 1)
        column, train_loss = epoch(nn, train_loader, "", False)
        loss_array.append(train_loss)
        print(f'- ---Epoch {e+1} \t\t Average Training Loss: {train_loss}\n')

    print("\n-------------------- VALIDATION LOOP -----------------------\n")

    min_valid_loss = np.inf
    val_array = [0] * int(epochs) 
    location = [0]*num_hazards
    solution_matrix = np.zeros((num_hazards,num_hazards), dtype = int)
    with torch.no_grad():
        nn.eval()
        for i in range(epochs):
            model_id, valid_loader = nn_common.gen_training_detaset(i, num_hazards, num_covariates, num_intervals, True, 1)
            column, valid_loss = epoch(nn, valid_loader, location, True)
            solution_matrix[model_id, column] += 1
            val_array.append(valid_loss)
            print(f'----Epoch {i+1} Average Validation Loss: {valid_loss}\n')
            if min_valid_loss > valid_loss:
                print(f'------Validation Loss Decreased({min_valid_loss:.3f}--->{valid_loss:.3f}) \t Saving The Model------\n')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(nn.state_dict(), 'saved_model.pth')

    train_avg = moving_average(loss_array, 2)  
    val_avg = moving_average(val_array, 2)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.set_title('Training Loss vs Epoch')
    ax.plot(train_avg, color="red")
    #ax.savefig(f"TrainingLossvsEpoch.png")    #Can be consolidated by using a function, maybe called graphResults(loss_array, val_array).
    ax2.set_title('Validation Loss vs Epoch')
    ax2.plot(val_avg, color="red")
    #ax2.savefig(f"ValidationLossvsEpoch.png")
    total_chances = epochs * 2**num_covariates
    plt.savefig(f"{output_directory}/ValandTrain.png")
    plt.close()


    # plt.close()
    # for model in common.models[:num_hazards]:
    #     plt.title(f"{model} - Amount and Locations of Accurate Prediction")
    #     plt.hist(accumulators[model], edgecolor='black')
    #     plt.plot()
    #     plt.savefig(f"{output_directory}/{model} - Amount and Locations of Accurate Prediction")
    #     plt.close()
    percent_array = location
    for i in range(len(location)):
       percent_array[i] = location[i]/epochs
    cumulativesum = np.cumsum(percent_array)
    

    solution_matrix_df = pd.DataFrame(solution_matrix)
    solution_matrix_df.to_csv("4haz1cov300epochLR0.0005.csv")
    x_axis = ["1st", "2nd", "3rd", "4th"]
    ax1 = plt.subplot(1,1,1)
    plt.ylim([0,1])
    ax1.set_ylabel('Percentage of correct predictions')
    ax1.bar(x_axis,percent_array, edgecolor = "black", color = '#6090C0')
    ax2 = ax1.twinx()
    ax2.set_ylabel('CDF')
    ax2.plot(x_axis,cumulativesum, color = '#cf6a63', linewidth = 3)
    plt.savefig("4haz1cov300epochLR00005")
    plt.show()
    # plt.title("Model Predictions")
    # plt.ylabel('Percentage of correct predictions')
    # plt.bar(("1st","2nd", "3rd", "4th", "5th"), percent_array , edgecolor = 'black')
    # plt.plot()
    # plt.plot()
    # plt.savefig('CumulativeModelAccuracy')
    plt.close()
    
    # for model in common.models[:num_hazards]:
    #     print(f"{model} accuracy: {len(accumulators[model])/epochs}\n")
    # print(f"The location of the correct placements is {location}")



#(num_hazards , num_covariates , num_intervals , learning_rate , weight_decay , output_directory)
train_model(len(common.models), 1, 25, 0.0005, 1e-6,"sim1")
