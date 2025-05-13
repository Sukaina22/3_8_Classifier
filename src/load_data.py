from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np 

def load_digits_filtered(digit1 = 3, digit2 = 8, test_size = 0.2, random_state = 42):
    digits = load_digits()
    X = digits.data
    Y = digits.target

    mask = (Y == digit1) | (Y == digit2)
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    
    Y_binary = np.where(Y_filtered == digit2, 1, 0)

    return train_test_split(X_filtered, Y_binary, test_size=test_size, random_state=random_state, stratify=Y_binary)