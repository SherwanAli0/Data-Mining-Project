import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Code"))

from preprocessing import INVALID_ZERO_COLUMNS, mark_invalid_zeros, impute_with_train_means


def make_frame(glucose, insulin):
    n = len(glucose)
    return pd.DataFrame({
        "Pregnancies": [0] * n,
        "Glucose": glucose,
        "BloodPressure": [70] * n,
        "SkinThickness": [20] * n,
        "Insulin": insulin,
        "BMI": [30.0] * n,
        "DiabetesPedigreeFunction": [0.5] * n,
        "Age": [40] * n,
    })


def test_mark_invalid_zeros_only_touches_listed_columns_and_keeps_input():
    X = make_frame([0, 100], [0, 80])
    marked = mark_invalid_zeros(X)
    assert marked["Glucose"].isna().tolist() == [True, False]
    assert marked["Insulin"].isna().tolist() == [True, False]
    assert (marked["Pregnancies"] == 0).all()  # a real zero, not a missing value
    assert (X["Glucose"] == [0, 100]).all()  # input not modified


def test_train_means_do_not_depend_on_heldout_rows():
    X_train = mark_invalid_zeros(make_frame([100, 120, 0, 140], [50, 0, 70, 90]))
    test_a = mark_invalid_zeros(make_frame([0, 110], [0, 60]))
    test_b = mark_invalid_zeros(make_frame([0, 999], [0, 5000]))  # only held-out values differ

    train_a, _, means_a = impute_with_train_means(X_train, test_a)
    train_b, _, means_b = impute_with_train_means(X_train, test_b)

    pd.testing.assert_series_equal(means_a, means_b)
    pd.testing.assert_frame_equal(train_a, train_b)
    assert means_a["Glucose"] == np.mean([100, 120, 140])
    assert means_a["Insulin"] == np.mean([50, 70, 90])


def test_missing_test_values_get_the_training_mean():
    X_train = mark_invalid_zeros(make_frame([100, 120, 140, 160], [50, 70, 90, 110]))
    X_test = mark_invalid_zeros(make_frame([0, 5000], [0, 9000]))
    _, test_filled, means = impute_with_train_means(X_train, X_test)
    assert test_filled["Glucose"].iloc[0] == means["Glucose"] == 130
    assert test_filled["Insulin"].iloc[0] == means["Insulin"] == 80
    assert test_filled["Glucose"].iloc[1] == 5000  # observed values are left alone
    assert not test_filled[INVALID_ZERO_COLUMNS].isna().any().any()


def test_committed_dataset_uses_training_rows_only():
    data = pd.read_csv(os.path.join(ROOT, "DataSet", "diabetes.csv"))
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, means = impute_with_train_means(mark_invalid_zeros(X_train), mark_invalid_zeros(X_test))

    expected = X_train[INVALID_ZERO_COLUMNS].replace(0, np.nan).mean()
    pd.testing.assert_series_equal(means, expected)

    # the old behaviour (means over all 768 rows) gives different values
    full_data_means = X[INVALID_ZERO_COLUMNS].replace(0, np.nan).mean()
    assert not np.allclose(means.values, full_data_means.values)
