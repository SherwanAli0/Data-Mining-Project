import pandas as pd
import numpy as np

# In the Pima dataset a 0 in these columns is not a real measurement.
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def mark_invalid_zeros(X, columns=INVALID_ZERO_COLUMNS):
    """Replace the invalid zeros with NaN. This is a fixed rule, nothing is learned from the data."""
    X = X.copy()
    X[columns] = X[columns].replace(0, np.nan)
    return X


def impute_with_train_means(X_train, X_test, columns=INVALID_ZERO_COLUMNS):
    """Fill NaN with column means computed from the training rows only.

    The same training means are used for the test rows, so held-out values
    never influence what is learned.
    """
    train_means = X_train[columns].mean()
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    X_train_filled[columns] = X_train_filled[columns].fillna(train_means)
    X_test_filled[columns] = X_test_filled[columns].fillna(train_means)
    return X_train_filled, X_test_filled, train_means
