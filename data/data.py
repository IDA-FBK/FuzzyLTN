import numpy as np
import pandas as pd  # Import pandas to load and process the CSV file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def get_data(dataset, seed=0, test_size=0.3):
    """
    Get data of the required dataset

    Parameters:
    - dataset (str): Name of the dataset.
    - seed (int): Seed for random number generation (used only for synthetic exp).
    - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    - data_train (tuple): Tuple containing features and labels for the training set.
    - data_test (tuple): Tuple containing features and labels for the testing set.
    """

    if dataset == "synthetic":
        rng = np.random.default_rng(seed)
        # Synthetic dataset generation code here

    elif dataset == "iris":
        # Load the Iris dataset
        iris = load_iris()
        x = iris.data
        y = iris.target

        # Convert to binary classification problem
        y[y == 0] = -1
        y[y > 0] = 1

        # Data normalization
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_normalized, y.reshape(-1, 1), test_size=test_size, random_state=seed
        )

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)

    elif dataset == "diabetes":
        # Assuming the diabetes dataset is stored in 'diabetes.csv'
        file_path = "data/diabetes.csv"

        # Lendo o arquivo
        df = pd.read_csv(file_path)
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        # Data normalization
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_normalized, y.reshape(-1, 1), test_size=test_size, random_state=seed
        )

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
    elif dataset == "liver":
        # Assuming the diabetes dataset is stored in 'liver_data.txt'
        file_path = "data/liver_data.txt"

        # Lendo o arquivo
        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable
        y[y == 2] = -1

        # Data normalization
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_normalized, y.reshape(-1, 1), test_size=test_size, random_state=seed
        )

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)
    elif dataset == "mammographic":
        # Assuming the diabetes dataset is stored in 'data/mammographic_masses.txt'
        file_path = "data/mammographic_masses.txt"

        # Lendo o arquivo
        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        # Data normalization
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_normalized, y.reshape(-1, 1), test_size=test_size, random_state=seed
        )

        data_train = (x_train, y_train)
        data_test = (x_test, y_test)

    return data_train, data_test
