"""
PreprocessData Script

- This script reads a CSV file containing movie data, performs preprocessing, and splits the data into training, testing, and validation sets. It uses the MultilabelStratifiedShuffleSplit from iterstrat.ml_stratifiers to ensure stratified splitting of the multilabel targets.
- The script assumes that the CSV file ("movies_genre.csv") is present in the "Data" directory and contains a column named 'overview' for movie overviews and multiple columns for binary genre labels.
- The resulting preprocessed data is saved in the "Data/Training", "Data/Testing", and "Data/Validation" directories as NumPy arrays.

Usage:
1. Ensure the "movies_genre.csv" file is present in the "Data" directory.
2. Run the script.

Example:
    python preprocess_data.py

Note: Make sure to install the required dependencies before running the script, including pandas, numpy, and iterstrat.

Author: Prashant Tiwari

Date: 20-12-2023
"""
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def PreprocessData():
    data = pd.read_csv("Data/movies_genre.csv")
    data.dropna(subset=['overview'], inplace=True)

    x = np.array(data['overview'])
    y = np.array(data.iloc[:, 3:])

    splits = MultilabelStratifiedShuffleSplit(2, test_size=0.25, random_state=1)
    for train_index, test_index in splits.split(x,y):
        x_train, y_train = x[train_index], y[train_index]
        x_testval, y_testval = x[test_index], y[test_index]

    splits = MultilabelStratifiedShuffleSplit(2, test_size=0.4, random_state=1)
    for test_index, val_index in splits.split(x_testval, y_testval):
        x_test, y_test = x_testval[test_index], y_testval[test_index]
        x_val, y_val = x_testval[val_index], y_testval[val_index]

    np.save("Data/Training/x_train.npy", x_train)
    np.save("Data/Training/y_train.npy", y_train)
    np.save("Data/Testing/x_test.npy", x_test)
    np.save("Data/Testing/y_test.npy", y_test)
    np.save("Data/Validation/x_val.npy", x_val)
    np.save("Data/Validation/y_val.npy", y_val)

if __name__ == '__main__':
    PreprocessData()