
import numpy as np
import pandas as pd

from NN import Neural_Network
from algo.score import Score
from algo.train import Train
from weight.weights import Weights

trainData = pd.read_csv("data/train.csv", header=None)
####ADDing commandline PArser"
"""parser = CommandLineParser()
parser.add_argument()"""


# Initializing all the parameters

batchsize = 5
momentum = 0.7
regularization = 0.00025
iteration = 5

depData = trainData.iloc[:, :2]
indepData = trainData.iloc[:, 2:]

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

offers = ['Offer_1','Offer_2]
scores = ['2', '1']

score = Score()

output, error = score.scoreData(weights, indepData,layers)
output = pd.concat([pd.DataFrame(depData), pd.DataFrame(output[2])], axis=1)
output.columns = offers + scores
filename = 'Insample_' + str(batchsize) + '_g' + '.csv'
output.to_csv(filename)

# ##Outsample_Scoring

# In[ ]:

validData = pd.read_csv("data/validation_ger.csv", header=None)
depData = validData.iloc[:, :2]
indepData = validData.iloc[:, 2:]
output, error = score.scoreData(weights, indepData,layers)
output = pd.concat([pd.DataFrame(depData), pd.DataFrame(output[2])], axis=1)
output.columns = offers + scores

filename = 'Outsample_' + str(batchsize) + '_g' + '.csv'
output.to_csv(filename)
