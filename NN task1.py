import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"C:\Users\User\Downloads\Lab2\NN-Task-1-perceptron\birds.csv")

# preprocessing .
genderMode = data['gender'].mode()
#print(genderMode.values) 

# number of males = number of females ,so fill by mode based on bird category.
data['gender'] = data.groupby('bird category')['gender'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None))

data['gender'] = data['gender'].apply(lambda x: 1 if x == 'male' else 0)

scaler = MinMaxScaler()

# numeric cols. for normalization
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# apply normalization
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])


#print(data)




def selectFeaturesFromGUI(feature1, feature2, class_selection):
    # data = pd.read_csv(csv_file)
    global data
    # Map for the selected classes
    class_map = {
        "A & B": ["A", "B"],
        "A & C": ["A", "C"],
        "B & C": ["B", "C"]
    }

    selected_classes = class_map.get(class_selection)
    
    # put selected bird categories in filtered data.
    filtered_data = data[data['bird category'].isin(selected_classes)]
    
    # put f1 , f2 in filtered data.
    filtered_data = filtered_data[[feature1, feature2, 'bird category']]

    # map the two classes to -1 , 1 .
    filtered_data['bird category'] = filtered_data['bird category'].apply(
        lambda x: -1 if x == selected_classes[0] else 1)

    return filtered_data


def perceptron_train(data, learning_rate, epochs, add_bias):
    weights = np.random.rand(2) * 0.01 #generate 2 random nums scaled to small num.
    bias = np.random.rand(1)[0] * 0.01 if add_bias else 0

    for epoch in range(epochs):
        for _, row in data.iterrows(): #loop on each row in data .
            x = np.array([row[feature1_var.get()], row[feature2_var.get()]])
            target = row['bird category']
            linear_output = np.dot(weights, x) + bias
            # print(linear_output)
            prediction = 1 if linear_output > 0 else -1 #signum function .
            if prediction != target:
                error = target - prediction
                weights += learning_rate * error * x
                if add_bias:
                    bias += learning_rate * error

    return weights, bias


def train_test_split(data):
    
    class_0 = data[data['bird category'] == -1]
    class_1 = data[data['bird category'] == 1]

    train = pd.concat([class_0.iloc[:30], class_1.iloc[:30]])
    #print(train)
    train = train.sample(frac=1).reset_index(drop=True) # shuffle data
    #print(train)
    test = pd.concat([class_0.iloc[30:], class_1.iloc[30:]])
    test = test.sample(frac=1).reset_index(drop=True) # shuffle data

    return train, test


def evaluate_perceptron(test_data, weights, bias):
    tp, fp, tn, fn = 0, 0, 0, 0
    for _, row in test_data.iterrows():
        x = np.array([row[feature1_var.get()], row[feature2_var.get()]])
        target = row['bird category']
        prediction = 1 if np.dot(weights, x) + bias > 0 else -1
        if prediction == target == 1:
            tp += 1
        elif prediction == 1 and target == -1:
            fp += 1
        elif prediction == target == -1:
            tn += 1
        elif prediction == -1 and target == 1:
            fn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(f"Confusion Matrix:\nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy: {accuracy:.2%}")


def plot_decision_boundary(data, weights, bias):
    x_vals = np.linspace(data[feature1_var.get()].min(), data[feature1_var.get()].max(), 100)
    y_vals = -(weights[0] * x_vals + bias) / weights[1]

    plt.figure()
    plt.scatter(data[feature1_var.get()][data['bird category'] == -1],
                data[feature2_var.get()][data['bird category'] == -1],
                label="Class -1", color="blue")
    plt.scatter(data[feature1_var.get()][data['bird category'] == 1],
                data[feature2_var.get()][data['bird category'] == 1],
                label="Class 1", color="orange")

    plt.plot(x_vals, y_vals, label="Decision Boundary", color="red")
    plt.xlabel(feature1_var.get())
    plt.ylabel(feature2_var.get())
    plt.legend()
    plt.show()


def adaline(data, learning_rate, epochs, add_bias, mse_threshold):
    weights = np.random.rand(2) * 0.01  # For two features
    bias = np.random.rand(1)[0] * 0.01 if add_bias else 0

    for epoch in range(epochs):
        epoch_errors = []
        for _, row in data.iterrows():
            x = np.array([row[feature1_var.get()], row[feature2_var.get()]])
            target = row['bird category']
            linear_output = np.dot(weights, x) + bias
            # print(linear_output)
            error = target - linear_output
            weights += learning_rate * error * x
            if add_bias:
                bias += learning_rate * error
            epoch_errors.append(0.5 * (error ** 2))

        mse = np.mean(epoch_errors)
        if mse < mse_threshold:
            print(f"Stopping early at epoch {epoch + 1} with MSE: {mse}")
            break
    return weights, bias


def evaluteAdaline(test_data, weights, bias):
    tp, fp, tn, fn = 0, 0, 0, 0
    for _, row in test_data.iterrows():
        x = np.array([row[feature1_var.get()], row[feature2_var.get()]])
        target = row['bird category']
        prediction = 1 if np.dot(weights, x) + bias > 0 else -1
        if prediction == target == 1:
            tp += 1
        elif prediction == 1 and target == -1:
            fp += 1
        elif prediction == target == -1:
            tn += 1
        elif prediction == -1 and target == 1:
            fn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(f"Confusion Matrix:\nTP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy: {accuracy:.2%}")



def start_perceptron():
    # Get user inputs
    selected_feature1 = feature1_var.get()
    selected_feature2 = feature2_var.get()
    selected_classes = classes_var.get()
    learning_rate = float(learning_rate_var.get())
    epochs = int(epochs_var.get())
    add_bias = bias_var.get()
    algorithm = algorithm_var.get()

    print(f"Features: {selected_feature1}, {selected_feature2}")
    print(f"Classes: {selected_classes}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Add Bias: {add_bias}")
   

    x = selectFeaturesFromGUI(selected_feature1, selected_feature2, selected_classes)
    tr, te = train_test_split(x)
    # print("train-------------->", tr)
    w, bias = perceptron_train(tr, learning_rate, epochs, add_bias)
    evaluate_perceptron(te, w, bias)
    plot_decision_boundary(tr, w, bias)
    # print(tr)
    # print(x)


def start_adaline():
    # Get user inputs
    selected_feature1 = feature1_var.get()
    selected_feature2 = feature2_var.get()
    selected_classes = classes_var.get()
    learning_rate = float(learning_rate_var.get())
    mse_threshold = float(mse_threshold_var.get())
    add_bias = bias_var.get()
    algorithm = algorithm_var.get()

    print(f"Features: {selected_feature1}, {selected_feature2}")
    print(f"Classes: {selected_classes}")
    print(f"Learning Rate: {learning_rate}")
    print(f"MSE Threshold: {mse_threshold}")
    print(f"Add Bias: {add_bias}")
  

    x = selectFeaturesFromGUI(selected_feature1, selected_feature2, selected_classes)
    tr, te = train_test_split(x)
    # print("train-------------->", tr)
    weights, bias = adaline(tr, learning_rate, 10000, add_bias, mse_threshold)
    evaluteAdaline(te, weights, bias)
   #print("Weights-------------->", weights)
    #print("Bias-------------->", bias)
    # plot_decision_boundary_adaline(weights, bias, tr, selected_feature1, selected_feature2)
    plot_decision_boundary(tr, weights, bias)
    # print(tr)
    # print(x)


root = tk.Tk()
root.title("Bird Classification Task")


def show_frame(frame):
    frame.tkraise()


frame_perceptron = tk.Frame(root)
frame_perceptron.grid(row=0, column=0, sticky="nsew")
frame_adaline = tk.Frame(root)
frame_adaline.grid(row=0, column=0, sticky="nsew")
main_menu = tk.Frame(root)
main_menu.grid(row=0, column=0, sticky="nsew")

# Buttons in main menu
btn_perceptron = tk.Button(main_menu, text="Single Perceptron", command=lambda: show_frame(frame_perceptron))
btn_perceptron.grid(row=0, column=0, padx=20, pady=20)

btn_adaline = tk.Button(main_menu, text="Adaline", command=lambda: show_frame(frame_adaline))
btn_adaline.grid(row=1, column=0, padx=20, pady=20)

# Variables for user input
feature1_var = tk.StringVar()
feature2_var = tk.StringVar()
classes_var = tk.StringVar()
learning_rate_var = tk.StringVar()
epochs_var = tk.StringVar()
mse_threshold_var = tk.StringVar()
bias_var = tk.BooleanVar()
algorithm_var = tk.StringVar()

# User input fields
tk.Label(frame_perceptron, text="Select Feature 1:").grid(row=0, column=0, padx=10, pady=5)
feature1_combobox = ttk.Combobox(frame_perceptron, textvariable=feature1_var,
                                 values=["gender", "body_mass", "beak_length", "beak_depth", "fin_length"])
feature1_combobox.grid(row=0, column=1, padx=10, pady=5)

tk.Label(frame_perceptron, text="Select Feature 2:").grid(row=1, column=0, padx=10, pady=5)
feature2_combobox = ttk.Combobox(frame_perceptron, textvariable=feature2_var,
                                 values=["gender", "body_mass", "beak_length", "beak_depth", "fin_length"])
feature2_combobox.grid(row=1, column=1, padx=10, pady=5)

tk.Label(frame_perceptron, text="Select Classes:").grid(row=2, column=0, padx=10, pady=5)
classes_combobox = ttk.Combobox(frame_perceptron, textvariable=classes_var, values=["A & B", "A & C", "B & C"])
classes_combobox.grid(row=2, column=1, padx=10, pady=5)

tk.Label(frame_perceptron, text="Learning Rate (eta):").grid(row=3, column=0, padx=10, pady=5)
tk.Entry(frame_perceptron, textvariable=learning_rate_var).grid(row=3, column=1, padx=10, pady=5)

tk.Label(frame_perceptron, text="Number of Epochs (m):").grid(row=4, column=0, padx=10, pady=5)
tk.Entry(frame_perceptron, textvariable=epochs_var).grid(row=4, column=1, padx=10, pady=5)

tk.Checkbutton(frame_perceptron, text="Add Bias", variable=bias_var).grid(row=6, columnspan=2, padx=10, pady=5)

tk.Label(frame_adaline, text="Select Feature 1:").grid(row=0, column=0, padx=10, pady=5)
feature1_combobox = ttk.Combobox(frame_adaline, textvariable=feature1_var,
                                 values=["gender", "body_mass", "beak_length", "beak_depth", "fin_length"])
feature1_combobox.grid(row=0, column=1, padx=10, pady=5)

tk.Label(frame_adaline, text="Select Feature 2:").grid(row=1, column=0, padx=10, pady=5)
feature2_combobox = ttk.Combobox(frame_adaline, textvariable=feature2_var,
                                 values=["gender", "body_mass", "beak_length", "beak_depth", "fin_length"])
feature2_combobox.grid(row=1, column=1, padx=10, pady=5)

tk.Label(frame_adaline, text="Select Classes:").grid(row=2, column=0, padx=10, pady=5)
classes_combobox = ttk.Combobox(frame_adaline, textvariable=classes_var, values=["A & B", "A & C", "B & C"])
classes_combobox.grid(row=2, column=1, padx=10, pady=5)

tk.Label(frame_adaline, text="Learning Rate (eta):").grid(row=3, column=0, padx=10, pady=5)
tk.Entry(frame_adaline, textvariable=learning_rate_var).grid(row=3, column=1, padx=10, pady=5)

tk.Label(frame_adaline, text="MSE Threshold:").grid(row=5, column=0, padx=10, pady=5)
tk.Entry(frame_adaline, textvariable=mse_threshold_var).grid(row=5, column=1, padx=10, pady=5)

tk.Checkbutton(frame_adaline, text="Add Bias", variable=bias_var).grid(row=6, columnspan=2, padx=10, pady=5)

# Button to start classification
start_button = tk.Button(frame_perceptron, text="Start Classification", command=start_perceptron)
start_button.grid(row=12, column=1, pady=10)
btn_back = tk.Button(frame_perceptron, text="Back to Main Menu", command=lambda: show_frame(main_menu))
btn_back.grid(row=12, column=0, padx=20, pady=10)

start_button = tk.Button(frame_adaline, text="Start Classification", command=start_adaline)
start_button.grid(row=12, column=1, pady=10)

btn_back = tk.Button(frame_adaline, text="Back to Main Menu", command=lambda: show_frame(main_menu))
btn_back.grid(row=12, column=0, padx=20, pady=10)

# Run the main window loop
root.mainloop()
