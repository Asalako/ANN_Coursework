import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plt


class Mlp(object):

    def __init__(self, data_set):
        # parameters
        # Classifier range value used to classifier points based on model
        self.classifier_range = self.classifier(data_set)
        self.learningRate = 0.01

        # Parameters to set the layers in the network
        network = {
            "inputNodes": 5,
            "hiddenNodes": 5,
            "outputNodes": 1
        }

        # Dictates a range of initial weight values based number of layer inputs/nodes
        self.layer1_size = network["inputNodes"] / 2
        self.layer2_size = network["hiddenNodes"] / 2

        # Initialising random weights in matrix. size is based on values in the network dictionary
        self.layer1_weights = np.random.uniform(-self.layer1_size, self.layer1_size,
                                                (network["inputNodes"], network["hiddenNodes"]))
        self.layer2_weights = np.random.uniform(-self.layer2_size, self.layer2_size,
                                                (network["hiddenNodes"], network["outputNodes"]))

        self.validation_history = []
        self.error_history = []
        self.momentum_bool = False
        self.guess_correct = 0
        self.models = []
        self.validation_count = 0

    # Forward propagation through the network returns activations values x layer weights
    def forward_pass(self, input_set):
        self.Sj = np.dot(input_set, self.layer1_weights)  # Dot product of input_set and first set of weights
        self.uj = self.sigmoid(self.Sj)
        self.uj_layer2 = np.dot(self.uj, self.layer2_weights)  # Dot product of hidden layer and second set of weights
        output = self.sigmoid(self.uj_layer2)
        return output

    # Activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivation of activation function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Backward propagation through the network and updates the weights
    def backward_pass(self, input_set, desired_output, output):
        # backward propagation through the network
        self.output_error = desired_output.T - output  # Error in output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        # Error in hidden layer
        self.hidden_error = self.output_delta.dot(
            self.layer2_weights.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.uj)

        # Making a copy of the weights for momentum calculations
        prev_w1 = np.copy(self.layer1_weights)
        prev_w2 = np.copy(self.layer2_weights)
        previous_weights = [prev_w1, prev_w2]

        # Updating weights
        self.layer1_weights += input_set.T.dot(self.hidden_delta) * self.learningRate
        self.layer2_weights += self.uj.T.dot(self.output_delta) * self.learningRate

        # Applies momentum calculation if bool is true
        if self.momentum_bool:
            self.momentum(previous_weights)

        # MSE output, was used for graphs
        #print("MSE:", self.mse(self.output_error))
        self.error_history.append(np.average(np.abs(self.mse(self.output_error))))

    # Applies forward and backward propagation to train the network
    def train(self, input_set, desired_output):
        output = self.forward_pass(input_set)
        self.backward_pass(input_set, desired_output, output)

    # Applies a forward propagation to the model
    def test_model(self, input_set, desired_output):
        desired_output = np.array([desired_output])
        actual_output = self.forward_pass(input_set)

        # Checks each data point to see if it fits in the model and updates a counter
        for i in range(len(actual_output)):
            result = self.class_point(desired_output.T[i], actual_output[i])
            # print("Model Output:", desired_output.T[i], "Observed Output: ", actual_output[i], result)

        # Calculating accuracy of the model
        self.correct = self.guess_correct
        self.accuracy = (self.correct / len(actual_output)) * 100
        self.guess_correct = 0
        #print("Accuracy: {:.3f}%".format(self.accuracy), self.correct)

        # return actual_output, desired_output.T

    # Calculating the loss using Mean Squared Error
    def mse(self, error):
        mse = (np.sum(error) ** 2) / len(error)
        return mse

        return mse

    # Mean Square Root Error
    def msre(self, error, output_set):
        sum = 0
        err_length = len(error)
        for i in range(err_length):
            sum += ((output_set[0][i] - error[i][0]) / error[i][0]) ** 2
        msre = sum / err_length
        return msre

    # Coefficient of Efficiency
    def ce(self, error, output_set):
        numerator = np.sum(error ** 2)
        mean = np.sum(output_set) / len(output_set)
        mean_arr = np.array([[mean] * len(output_set)])
        denominator = np.sum((output_set - mean_arr) ** 2)
        ce = 1 - (numerator / denominator)
        return ce

    # Validation on the model. Will terminate the training if the model seems to be degrade 
    def validate(self, input_set, desired_output, epoch):
        desired_output = np.array([desired_output])
        actual_output = self.forward_pass(input_set)
        output_error = actual_output - desired_output.T
        validation_mse = self.mse(output_error)

        # Check to make sure the list is not empty
        if len(self.validation_history) > 1:
            prev_mse = self.validation_history[-1]
            # Compares previous error and current, append model
            if validation_mse > prev_mse:
                self.validation_count += 1
                self.models.append([np.copy(self.layer1_weights), np.copy(self.layer2_weights), epoch])
                # Condition to stop model continuously degrade
                if self.validation_count >= 3 and difference_percentage(validation_mse, prev_mse) < 0.001:
                    self.validation_count = 0
                    return True
            else:
                self.validation_count = 0

        self.validation_history.append(validation_mse)
        return False

    # Applies momentum calculation to the weights
    def momentum(self, prev_weights):
        constant = 0.9
        self.layer1_weights += (self.layer1_weights - prev_weights[0]) * constant
        self.layer2_weights += (self.layer2_weights - prev_weights[1]) * constant

    # Applies annealing to the learning rate which modifies the amount the weights as the model is trained
    def annealing(self, max_epochs, epoch, input_set, desired_output, start=0.05, end=0.01):
        end_lr = end
        start_lr = start
        anneal = end_lr + (start_lr - end_lr) * (1 - (1 / (1 + math.exp(10 - ((20 * epoch) / max_epochs)))))
        self.learningRate = anneal
        self.train(input_set, desired_output)

    # Gets classifier based of avg difference between the possible standardisation values multiplied by 2
    def classifier(self, predictand):
        classes = []
        # Gets unique standardisation values and sorts them
        for i in predictand:
            if i not in classes:
                classes.append(float(i))
        classes = sorted(classes)

        # Finds avg between each standardisation value
        sum = 0
        for i in range(len(classes) - 1):
            sum += abs(classes[i + 1] - classes[i])
        avg = sum * 2 / len(classes)
        return avg

    # Incrementally counts the number of data points that fit the model
    def class_point(self, model, observed):
        diff = abs(model - observed)
        if diff < self.classifier_range:
            self.guess_correct += 1
            return "True"
        return "False"


# Converts data from a spreadsheet which is stored as a dictionary to an list
def dict_to_list(data, columns):
    li = []
    # Scenario for when dictionary only has one key
    if len(columns) == 1:
        for row in range(len(data)):
            value = data.iloc[row][columns[0]]
            li.append(value)
        return li

    # Scenario for when dictionary has multiple keys
    for row in range(len(data)):
        record = []
        for column in columns:
            record.append(data.iloc[row][column])
        li.append(record)
    return li


# Standardises the given value
def standardisation(input_data, minimum, maximum):
    s = 0.8 * ((input_data - minimum) / (maximum - minimum)) + 0.1
    return round(s, 3)


# Standardises the data structure
def standardise_dataset(data, columnNames):
    data_set = pd.DataFrame(data, columns=columnNames)
    data_dict = data_set.to_dict()
    for key in data_set:
        predictor_name = data_dict[key]
        minimum = data_set[key].min()
        maximum = data_set[key].max()

        for key in predictor_name:
            s = standardisation(predictor_name[key], minimum, maximum)
            predictor_name[key] = s

    return pd.DataFrame(data_dict)


def graph(epochs, NN):
    plt.plot(range(epochs+1), NN.error_history)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.show()


def difference_percentage(value1, value2):
    if value1 < value2:
        value2, value1 = value1, value2
    return (value1 - value2) / value1

# Getting data from spreadsheet
data = pd.read_excel(
    r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx')

columns = ["T", "W", "SR", "DSP", "DRH", "PanE"]
predictor = ["T", "W", "SR", "DSP", "DRH"]
predictand = ["PanE"]
dict_to_list(data, columns)
dataset = standardise_dataset(data, columns)

# Splitting data into training, validation and testing sets
training_set = dataset.sample(frac=0.6, replace=False)
train_input = dict_to_list(training_set, predictor)
train_output = dict_to_list(training_set, predictand)
train_input = np.array(train_input)
train_output = np.array([train_output])

validation_set = dataset.sample(frac=0.2, replace=False)
valid_input = dict_to_list(validation_set, predictor)
valid_output = dict_to_list(validation_set, predictand)

testing_set = dataset.sample(frac=0.2, replace=False)
test_input = dict_to_list(testing_set, predictor)
test_output = dict_to_list(testing_set, predictand)

NN = Mlp(dict_to_list(dataset, predictand))
epochs = 10000
for i in range(epochs):  # trains the NN 1000 times
    NN.train(train_input, train_output)
    # NN.annealing(epochs, i, train_input, train_output)
    if i % 5 == 0:
        result = NN.validate(valid_input, valid_output, i)
        if result:
            graph(i, NN)
            break

acc = 0
if result:
    for model in NN.models:
        NN.layer1_weights = model[0]
        NN.layer2_weights = model[1]
        NN.test_model(test_input, test_output)
        if NN.accuracy > acc:
            acc = NN.accuracy
            best_model = [model, NN.accuracy]
        graph(i, NN)

else:
    NN.test_model(test_input, test_output)
    best_model = [[NN.layer1_weights, NN.layer2_weights, i], NN.accuracy]
    graph(i, NN)

print(best_model[0][0])
print(best_model[0][1])
print("Accuracy: {:.3f}%".format(best_model[1]))
print("Total Epochs:", best_model[0][2])

# Plotting graph
