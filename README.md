
# Project Documentation
## Overview
This project uses implementation of an Adaptive Neuro-Fuzzy Inference System (ANFIS) based on TensorFlow https://github.com/tiagoCuervo/TensorANFIS. The ANFIS model is a fuzzy logic-based system that adapts and refines its rules based on training data. In this implementation. Huber loss function and Adam optimizer are used in conjunction with Gaussian functions to model the fuzzy rules.
## Dataset
The dataset used in this project is the Wine Quality dataset, sourced from Kaggle, which, in turn, is based on the UCI Machine Learning Repository's Wine Quality dataset. The dataset includes physicochemical attributes of wines and a quality score ranging from 0 to 10.
Input Variables (Physicochemical Attributes):
Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free Sulfur Dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol
Output Variable:
Quality (Score between 0 and 10)

## Project Files
1. WineQT.csv
The dataset contains information about various wines, including physicochemical attributes and quality scores.
2. anfis.py
Contains the implementation of the ANFIS model based on Gaussian functions.
3. wineTrain.py
The main script for training the ANFIS model using standard training techniques.
4. wineTrainKFOLD.py
A script for training the ANFIS model using K-Fold cross-validation.
5. requirements.txt
A file specifying the required Python packages and their versions for this project.
Normalization
Data normalization is crucial for the model's performance due to the varying scales of input features. All input features and labels are normalized to ensure consistent scaling and improved model convergence. Both input features and labels are normalized using StandardScaler.
K-Fold Cross-Validation
An alternative training approach involves K-Fold cross-validation. The model is trained and evaluated K times, each time using a different fold as the validation set and the remaining folds as the training set. This approach helps assess the model's robustness and generalization across different subsets of the dataset.


## Training

Based on training loss, validation loss, and a few training trials, the following hyperparameters were used:
Standard Training (wineTrain.py)
Learning Rate: 0.01
Number of Rules (Gaussian Functions): 300
Number of Epochs: 200



Manual Prediction:  [7.00243996]
Actual Value:  7
K-Fold Cross-Validation (wineTrainKFOLD.py)
Learning Rate: 0.01
Number of Rules (Gaussian Functions): 300
Number of Epochs: 200


Final Training loss: 0.0134824
Final Validation loss:0.45183455
Single validation Manual Prediction:  [7.06378493]
Actual Value:  7
Manual Prediction
A manual prediction selected item (wine) is performed on a sample input with the model, demonstrating the process of unnormalizing the prediction for better interpretability.


## Conclusion
The project demonstrates the training and evaluation of an ANFIS model on the Wine Quality dataset, highlighting the importance of normalization and providing options for both standard training and K-Fold cross-validation, with very high precision:(validation loss: 0.439289 for standard training, validation loss: 0.451834 for K-Fold cross validation).
Implementation Details
The implementation is based on Gaussian functions, which are utilized to model the fuzzy rules within the ANFIS architecture.
Installation
Suggested installation steps include creating a conda environment:
bash
conda create -n tf15 python tensorflow=1.15

Next, install the required dependencies using the requirements.txt file:
bash
pip install -r requirements.txt

## Requirements
Known dependencies:
- Python (3.5.5)
- Tensorflow (1.15.2)
- Numpy (1.15.2)
- Matplotlib (3.0.0)
- SciKit-Learn (1.0.2)
To install dependencies, `cd` to the directory of the repository and run `pip install -r requirements.txt`
To run the example, `cd` to the directory of the repository and run `python wineTrain.py`