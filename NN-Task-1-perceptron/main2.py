import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv(r"G:\FCIS YEAR4,TERM1\Neural Networks\Tasks\Task 2\NN Task 2\birds.csv")


data['gender'] = data.groupby('bird category')['gender'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None))

data['gender'] = data['gender'].apply(lambda x: 1 if x == 'male' else 0)

data['bird category'] = LabelEncoder().fit_transform(data['bird category']) # 0,1,2 

scaler = MinMaxScaler()

numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
#print(data['bird category'].dtype) # int32


def initialize_network():
        hidden_layers = int(hidden_layers_entry.get())
        neurons = list(map(int, neurons_entry.get().split(',')))
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())
        add_bias = bias_var.get()
        activation_function = activation_var.get()
        #print(neurons)
        if len(neurons) != hidden_layers:
            raise ValueError("Number of neurons must match the number of hidden layers.")
        train,test=train_test_split(data)
        w,b=back_probagation(train,hidden_layers,neurons,learning_rate,epochs,add_bias,activation_function)
        evaluate_performance(test,w,b,activation_function)
        showinfo("Initialization", "Network parameters captured successfully!")
        

def train_test_split(data):
    train_set = []
    test_set = []

    for class_label in data['bird category'].unique():
        class_data = data[data['bird category'] == class_label]
        train_set.append(class_data.iloc[:30])
        test_set.append(class_data.iloc[30:])
    #print(len(test_set[0])+len(test_set[1])+len(test_set[2]))
    train_set = pd.concat(train_set).sample(frac=1).reset_index(drop=True)  # Shuffle training set
    test_set = pd.concat(test_set).sample(frac=1).reset_index(drop=True)    # Shuffle testing set
    

    return train_set, test_set

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    return x*(1-x)

def hyperbolic_tangent(x):
    return np.tanh(x)

def hyperbolic_tangent_derivative(x):
    return 1-(x*x)


def evaluate_performance(testData, weights, biases, activation):
    if activation == "Sigmoid":
        activation_function = sigmoid
    else:
        activation_function = hyperbolic_tangent

    target = testData.iloc[:, -1].to_numpy().astype(int)  
   
    confusion_matrix = np.zeros((3, 3), dtype=int) 

    for index, row in testData.iterrows():
        sample = row[:-1].to_numpy().reshape(1, -1)  
        true_label = int(target[index])  
        layer_outputs = [sample]
        for w, b in zip(weights, biases):
            z = np.dot(layer_outputs[-1], w) + b
            a = activation_function(z)
            layer_outputs.append(a)
        final_output = layer_outputs[-1]
        one_hot_output = np.zeros_like(final_output)
        one_hot_output[0, np.argmax(final_output)] = 1  
        predicted_class = np.argmax(one_hot_output)
        confusion_matrix[true_label, predicted_class] += 1 
        
    print("Test Confusion Matrix:")
    print(confusion_matrix)

    total_correct = np.trace(confusion_matrix)  
    total_predictions = np.sum(confusion_matrix)  
    TestAccuracy = total_correct / total_predictions if total_predictions != 0 else 0.0

    print(f"Test Accuracy: {TestAccuracy * 100:.2f}%")


def back_probagation(x,hidden_layers,neurons,learning_rate,epochs,add_bias,activation_function):
    weights = []
    biases = []
    target = x.iloc[:, -1].to_numpy().reshape(-1,1) 
    #print("target--------->",target)
    if activation_function == "Sigmoid":
        activation = sigmoid
        activation_derivative = sigmoid_derivative
    else: 
        activation = hyperbolic_tangent
        activation_derivative = hyperbolic_tangent_derivative
   
    # for input layer
    weights.append(np.random.randn(5, neurons[0]) * 0.01)
    biases.append(np.random.randn(1, neurons[0]) * 0.01 if add_bias else np.zeros((1, neurons[0])))
    #print("w[0]",weights[0])
    # for hidden layer
    for i in range(hidden_layers - 1):
        weights.append(np.random.randn(neurons[i], neurons[i + 1]) * 0.01)
        biases.append(np.random.randn(1, neurons[i + 1]) * 0.01 if add_bias else np.zeros((1, neurons[i + 1])))

    # for output
    weights.append(np.random.randn(neurons[-1], 3) * 0.01)
    biases.append(np.random.randn(1, 3) * 0.01 if add_bias else np.zeros((1, 3)))
    
    for i in range(epochs):
        for index, row in x.iterrows():
            sample = row[:-1].to_numpy().reshape(1, -1) 
            label = np.zeros((1, 3))            
            label[0, int(target[index].item())] = 1   #2D arr 1x3
            
            
            # forward pass
            layer_outputs = [sample]
            for w, b in zip(weights, biases):
                z = np.dot(layer_outputs[-1], w) + b
                a = activation(z)
                layer_outputs.append(a)
            
            # backward pass
            error_signal=layer_outputs[-1]-label
            #print("error signal------------>",error_signal)
            #print(layer_outputs[-1])
            for l in reversed(range(len(weights))): 
                if l == len(weights) - 1:  # for output layer
                    delta = error_signal * activation_derivative(layer_outputs[-1])
                else:  # for other layers
                    delta = np.dot(delta, weights[l + 1].T) * activation_derivative(layer_outputs[l + 1])
                    #print("weights",weights[l+1].shape)
                    #print("layer output delta",layer_outputs[l+1].shape)

                dW = np.dot(layer_outputs[l].T, delta)
                db = np.sum(delta, axis=0, keepdims=True)
                #print("dw shape----------->",dW.shape)
                weights[l] -= learning_rate * dW
                biases[l] -= learning_rate * db
                
        # calculate training accuracy each epoch
        confusion_matrix = np.zeros((3, 3), dtype=int) 
        for index, row in x.iterrows():
            sample = row[:-1].to_numpy().reshape(1, -1)  
            true_label = int(target[index].item())  

            
            layer_outputs = [sample]
            for w, b in zip(weights, biases):
                z = np.dot(layer_outputs[-1], w) + b
                a = activation(z)
                layer_outputs.append(a)

            
            final_output = layer_outputs[-1]
            one_hot_output = np.zeros_like(final_output)
            one_hot_output[0, np.argmax(final_output)] = 1  
            predicted_class = np.argmax(one_hot_output)

            
            confusion_matrix[true_label, predicted_class] += 1 
       
        total_correct = np.trace(confusion_matrix)  
        total_predictions = np.sum(confusion_matrix)  
        train_accuracy = total_correct / total_predictions if total_predictions != 0 else 0.0

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

    print("Train Confusion Matrix:")
    print(confusion_matrix)
    print("            ")
    return weights,biases
        
        
# Create the main window
root = tk.Tk()
root.title("Neural Network")

# User Input Section
ttk.Label(root, text="Number of Hidden Layers:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
hidden_layers_entry = ttk.Entry(root)
hidden_layers_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Neurons in Each Hidden Layer (comma-separated):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
neurons_entry = ttk.Entry(root)
neurons_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text="Learning Rate (η):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
learning_rate_entry = ttk.Entry(root)
learning_rate_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(root, text="Number of Epochs (m):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
epochs_entry = ttk.Entry(root)
epochs_entry.grid(row=3, column=1, padx=10, pady=5)

bias_var = tk.BooleanVar()
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")

activation_var = tk.StringVar(value="Sigmoid")
ttk.Label(root, text="Activation Function:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
ttk.Radiobutton(root, text="Sigmoid", variable=activation_var, value="Sigmoid").grid(row=5, column=1, padx=10, pady=5, sticky="w")
ttk.Radiobutton(root, text="Hyperbolic Tangent", variable=activation_var, value="Tanh").grid(row=6, column=1, padx=10, pady=5, sticky="w")

# Buttons
ttk.Button(root, text="Initialize", command=initialize_network).grid(row=7, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()

