import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from anfis import ANFIS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Load WineQT dataset (replace 'your_dataset.csv' with the actual filename)
wine_data = np.loadtxt('WineQT.csv', delimiter=',', skiprows=1)

allData = wine_data[:, :-2]  # Features
allLabels = wine_data[:, -2]  # Target variable

# Normalize the data
scaler = StandardScaler()
allData_normalized = scaler.fit_transform(allData)

scalerLabels = StandardScaler()
# Normalize the labels using the same scaler
allLabels_normalized = scalerLabels.fit_transform(allLabels.reshape(-1, 1)).flatten()

# ANFIS params and Tensorflow graph initialization
m = 300  # number of rules
alpha = 0.01  # learning rate

# Set up k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(allData_normalized)):
    tf.reset_default_graph()  # Reset the TensorFlow graph at the beginning of each fold

    trainData_fold, validationData_fold = allData_normalized[train_idx], allData_normalized[val_idx]
    trainLabels_fold, validationLabels_fold = allLabels_normalized[train_idx], allLabels_normalized[val_idx]

    fis = ANFIS(n_inputs=11, n_rules=m, learning_rate=alpha)

    # Training
    num_epochs = 200

    # Initialize session to make computations on the TensorFlow graph
    with tf.Session() as sess:
        # Initialize model parameters
        sess.run(fis.init_variables)
        trn_costs = []
        val_costs = []
        time_start = time.time()
        for epoch in range(num_epochs):
            # Run an update step
            trn_loss, trn_pred = fis.train(sess, trainData_fold, trainLabels_fold)
            # Evaluate on the validation set
            val_pred, val_loss = fis.infer(sess, validationData_fold, validationLabels_fold)
            if epoch % 10 == 0:
                print(f"Fold {fold + 1}, Epoch {epoch}, Train cost: {trn_loss}, Validation loss: {val_loss}")

            trn_costs.append(trn_loss)
            val_costs.append(val_loss)

        # Plot the cost over epochs for each fold
        plt.figure(fold + 1)
        plt.subplot(2, 1, 1)
        plt.plot(np.squeeze(trn_costs))
        plt.title(f"Fold {fold + 1} - Training loss, Learning rate = {alpha}")
        plt.subplot(2, 1, 2)
        plt.plot(np.squeeze(val_costs))
        plt.title(f"Fold {fold + 1} - Validation loss, Learning rate = {alpha}")
        plt.ylabel('Cost')
        plt.xlabel('Epochs')

        # Manual Prediction
        manual_input_values = [7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0]
        manual_input = np.array([manual_input_values]).astype(np.float32)
        manual_input_normalized = scaler.transform(manual_input)

        manual_prediction_normalized = fis.infer(sess, manual_input_normalized)
        # Unnormalize the predicted value
        manual_prediction = scalerLabels.inverse_transform([manual_prediction_normalized])[0]
        print("Manual Prediction: ", manual_prediction)
        print("Actual Value: ", 7)
# Show the plots for all folds
plt.show()
