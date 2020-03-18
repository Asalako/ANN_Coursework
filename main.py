
import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt

class Mlp(object):
    def __init__(self):
        # parameters
        self.size = 2.5
        self.inputSize = 5
        self.outputSize = 1
        self.hiddenSize = 2
        self.learningRate = 0.01
        network = {
            "inputNodes": 5,
            "hiddenNodes": 5,
            "outputNodes": 1
        }

        # weights
        self.layer1_weights = np.random.uniform(-self.size, self.size, (network["inputNodes"], network["hiddenNodes"]))
        self.layer2_weights = np.random.uniform(-self.size, self.size, (network["hiddenNodes"], network["outputNodes"]))

        self.error_history = []
        self.momentum_bool = False

    def feedForward(self, input_set):
        # forward propogation through the network
        self.Sj = np.dot(input_set, self.layer1_weights)  # dot product of input_set (input) and first set of weights (3x2)
        self.uj = self.sigmoid(self.Sj)  # activation function
        self.uj_layer2 = np.dot(self.uj, self.layer2_weights)  # dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.uj_layer2)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def backward(self, input_set, desired_output, output):
        # backward propogate through the network
        self.output_error = desired_output.T - output  # error in output
        self.output_delta = self.output_error * self.sigmoidDerivative(output)

        self.hidden_error = self.output_delta.dot(
            self.layer2_weights.T)  # z2 error: how much our hidden layer weights contribute to output error
        self.hidden_delta = self.hidden_error * self.sigmoidDerivative(self.uj)  # applying derivative of sigmoid to z2 error

        prev_w1 = np.copy(self.layer1_weights)
        prev_w2 = np.copy(self.layer2_weights)
        previous_weights = [prev_w1, prev_w2 ]

        self.layer1_weights += input_set.T.dot(self.hidden_delta) * self.learningRate  # adjusting first set (input -> hidden) weights
        self.layer2_weights += self.uj.T.dot(self.output_delta) * self.learningRate # adjusting second set (hidden -> output) weights

        if self.momentum_bool:
            self.momentum(previous_weights)

        ce_error = self.ce(self.output_error, output)
        self.error_history.append(np.average(np.abs(ce_error)))

        # print(self.mse(self.output_error))
    def train(self, input_set, desired_output):
        output = self.feedForward(input_set)
        self.backward(input_set, desired_output, output)

    def testModel(self, input_set, desired_output):
        desired_output = np.array([desired_output])
        actual_output = self.feedForward(input_set)
        output_error =  actual_output - desired_output.T
        return self.mse(output_error)

    def mse(self, error):
        mse = ((np.sum(error)**2)/len(error) )**1/2
        return mse

    def ce(self, error, output_set):
        numerator = np.sum(error**2)
        mean = np.sum(output_set) / len(output_set)
        mean_arr = np.array([[mean] * len(output_set)])
        denominator =  np.sum((output_set - mean_arr)**2)
        ce = 1 - (numerator / denominator)
        print(ce)
        return ce

    def validate(self, input_set, desired_output):
        desired_output = np.array([desired_output])
        actual_output = self.feedForward(input_set)
        output_error = actual_output - desired_output.T
        
    def momentum(self, prev_weights):
        constant = 0.9
        self.layer1_weights += (self.layer1_weights - prev_weights[0]) * constant
        self.layer2_weights += (self.layer2_weights - prev_weights[1]) * constant

    def annealing(self, max_epochs, epoch, input_set, desired_output, start=0.05, end=0.01):
        end_lr = end
        start_lr = start
        anneal = end_lr + ( start_lr - end_lr) * ( 1 - (1 / (1 + math.exp( 10 - ((20*epoch) / max_epochs) ))))
        self.learningRate = anneal
        self.train(input_set, desired_output)


def arrayCon(arr):
    array = [[elem] for elem in arr]
    return array


def dictToList(data, columns):
    li = []
    if len(columns) == 1:
        for row in range(len(data)):
            value = data.iloc[row][columns[0]]
            li.append(value)
        return li

    for row in range(len(data)):
        record = []
        for column in columns:
            record.append(data.iloc[row][column])
        li.append(record)
    return li


def standardisation(inputData, minimum, maximum):
    s = 0.8 * ((inputData - minimum) / (maximum - minimum)) + 0.1
    return round(s, 3)


def standardiseDataset(data, columnNames):
    dataset = pd.DataFrame(data, columns=columnNames)
    dataDict = dataset.to_dict()
    for key in dataset:
        predictor = dataDict[key]
        minimum = dataset[key].min()
        maximum = dataset[key].max()

        for key in predictor:
            s = standardisation(predictor[key], minimum, maximum)
            predictor[key] = s

    return pd.DataFrame(dataDict)

data = pd.read_excel(
    r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx')

columns = ["T", "W", "SR", "DSP", "DRH", "PanE"]
predictor = ["T", "W", "SR", "DSP", "DRH"]
predictand = ["PanE"]
dictToList(data, columns)
dataset = standardiseDataset(data, columns)

training_set = dataset.sample(frac=0.6, replace=False)
train_input = dictToList(training_set, predictor)
train_output = dictToList(training_set, predictand)
train_input = np.array(train_input)
train_output = np.array([train_output])

validation_set = dataset.sample(frac=0.2, replace=False)
valid_input = dictToList(validation_set, predictor)
valid_output = dictToList(validation_set, predictand)

testing_set =  dataset.sample(frac=0.2, replace=False)
test_input = dictToList(testing_set, predictor)
test_output = dictToList(testing_set, predictand)



NN = Mlp()
epochs = 30
for i in range(epochs): #trains the NN 1000 times
    # if (i % 100 == 0):
    #     print("Loss: " + str(np.mean(np.square(desired_output - NN.feedForward(input_set)))))
    # NN.train(train_input, train_output)
    NN.annealing(epochs, i, train_input, train_output)
print("test", NN.testModel(test_input, test_output))

# epochs = []
# results = []
# p = Mlp(train_input, train_output)
# for j in range(1):
#     p.trainNetwork()
    # p.output()
    #if j % 100 == 0:

plt.plot(range(epochs), NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Coefficient of Efficiency')
plt.show()

#
#     data = np.squeeze()
#     plt.plot(data)
#     plt.ylabel("MSE, LR")
#     plt.xlabel('Epochs. Hidden Nodes: ')
#     plt.show()
