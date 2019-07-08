# Neural-Networks
A Fully connected FF BP Neural Network Library built from Ground up using vanilla Python
The library has been built for a multi layer perceptron and the structure of the network can be tweaked 
by modifying the json file under data

A command line parser has been built to invoke the deep learning framework under parser/cli_parser.py . USe your own format_file.txt to create and build subparsers.

All the activation functions are kept in activation/

The Utils contains static methods for creation of zero and ones matrices.

The Weight folder has a initialized set of weights for testing.

The Error folder has custom defined Errors.


Lastly, the FFbackprop folder has the core algorithm and optimisers execution under FFbackprop/train and FFbackprop/score.


feel Free to reach out to me  for feedback.



You can implement a driver file to use this network:
 



### Sample Binary Classifier

import numpy as np
import pandas as pd

from struct_NN import Neural_Network
from FFbackprop.score import Score
from FFbackprop.train import Train
from weight.weights import Weights

trainData = pd.read_csv("data/train.csv", header=None)

parser = CommandLineParser()
parser.add_argument()

batchsize = 10,momentum = 0.5,regularization = 0.00025,iteration = 10

depData = "Dep rows"
indepData = "Indep rows"

weights_1 = "Retraining weights
weights_2 = "Retrainingf weights for layer 2
check = Neural_Network({1: ["Sigmoid", 5]}, ["CEE", "Sigmoid"], indepData, depData)
train = Train()
architecture = check.architecture
index = check.index
train.initParameters(regularization, momentum, batchsize, iteration,architecture)
layers = check.initlayer()
indep, dep = check.indepData, check.depData
weight = Weights()
weights = weight.initWeights({}, False, Weight1, Weight2)
weights = train.trainData(weights,index,indep,dep,layers)

validData = pd.read_csv("data/test.csv", header=None)
depData = "test data dep rows"
indepData = "test data indep rows"
output, error = score.scoreData(weights, indepData,layers)



