import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from anfis import ANFIS
from sklearn.preprocessing import StandardScaler

# Load WineQT dataset (replace 'your_dataset.csv' with the actual filename)
wine_data = np.loadtxt('WineQT.csv', delimiter=',', skiprows=1)

allData = wine_data[:, :-2]  # Features
allLabels = wine_data[:, -2]  # Target variable

# Split dataset into training and testing randomly (replace 0.7 by the desired split ratio)
np.random.seed(5)
indices = np.random.permutation(allData.shape[0])
training_idx, test_idx = indices[:int(0.7 * allData.shape[0])], indices[int(0.7 * allData.shape[0]):]
trainData, validationData = allData[training_idx, :], allData[test_idx, :]
trainLabels, validationLabels = allLabels[training_idx], allLabels[test_idx]

# Convert all data to float32
trainData = trainData.astype(np.float32)
trainLabels = trainLabels.astype(np.float32)
validationData = validationData.astype(np.float32)
validationLabels = validationLabels.astype(np.float32)

# Normalize the data
scaler = StandardScaler()
trainData_normalized = scaler.fit_transform(trainData)
validationData_normalized = scaler.transform(validationData)

scalerLabels = StandardScaler()
# Normalize the labels using the same scaler
trainLabels_normalized = scalerLabels.fit_transform(trainLabels.reshape(-1, 1)).flatten()
validationLabels_normalized = scalerLabels.transform(validationLabels.reshape(-1, 1)).flatten()

# ANFIS params and Tensorflow graph initialization
m = 300  # number of rules
alpha = 0.01  # learning rate
plt.plot(trainData_normalized)
plt.plot(validationData_normalized)

fis = ANFIS(n_inputs=11, n_rules=m, learning_rate=alpha)

# Training
num_epochs = 200

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        # Run an update step
        trn_loss, trn_pred = fis.train(sess, trainData_normalized, trainLabels_normalized)
        # Evaluate on validation set
        val_pred, val_loss = fis.infer(sess, validationData_normalized, validationLabels_normalized)
        if epoch % 10 == 0:
            print("Epoch %i, Train cost: %f, Validation loss: %f" %  (epoch, trn_loss, val_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)
    # Plot the cost over epochs
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(trn_costs))
    plt.title("Training loss, Learning rate =" + str(alpha))
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(val_costs))
    plt.title("Validation loss, Learning rate =" + str(alpha))
    plt.ylabel('Cost')
    plt.xlabel('Epochs')

    # Manual Prediction
    manual_input_values = [7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0]
    manual_input = np.array([manual_input_values]).astype(np.float32)
    manual_input_normalized = scaler.transform(manual_input)

    manual_prediction_normalized  = fis.infer(sess, manual_input_normalized)
    # Unnormalize the predicted value
    manual_prediction = scalerLabels.inverse_transform([manual_prediction_normalized])[0]
    print("Manual Prediction: ", manual_prediction)
    print("Actual Value: ", 7)

    plt.show()
