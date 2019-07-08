

"""Sample Binary Classification."""

import numpy as np
import pandas as pd

from NN import Neural_Network
from ffprop.score import Score
from ffprop.train import Train
from weight.weights import Weights

trainData = pd.read_csv("data/train.csv", header=None)

parser = CommandLineParser()
parser.add_argument()


# Initializing all the parameters

batchsize = 10
momentum = 0.5
regularization = 0.00025
iteration = 10

depData = trainData.iloc[:, :6]
indepData = trainData.iloc[:, 7:]

Weight1 = np.mat(pd.read_csv("data/weights1.csv", header=None).iloc[:, :5])
Weight2 = np.mat(pd.read_csv("data/Weights2.csv", header=None).iloc[:, :2])

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

# ##Insample_Scoring

# In[ ]:

classes = ['Class_1','Class_2]

score = Score()

output, error = score.scoreData(weights, indepData,layers)
output = pd.concat([pd.DataFrame(depData), pd.DataFrame(output[2])], axis=1)
output.columns = offers 
# filename = 'Insample_' + str(batchsize) + '_g' + '.csv'
# output.to_csv(filename)

# ##Outsample_Scoring

# In[ ]:

validData = pd.read_csv("data/test.csv", header=None)
depData = validData.iloc[:, :2]
indepData = validData.iloc[:, 2:]
output, error = score.scoreData(weights, indepData,layers)
output = pd.concat([pd.DataFrame(depData), pd.DataFrame(output[2])], axis=1)
output.columns = offers

# filename = 'Outsample_' + str(batchsize) + '_g' + '.csv'
# output.to_csv(filename)
