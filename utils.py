import os
import pickle
import pandas as pd

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


def load_dataset(file):
    # Load data into pandas dataframe
    data = pd.read_csv(file)

    # Only use first 5 Categories:
    df = data.iloc[:, : 5]

    return df


def create_train_test_set(data, visualize=False):
    # Training data only for JAHR < 2020
    train_data = data[data["ds"] <= '2019-12-01']
    test_data = data[data["ds"] > '2019-12-01']

    print(f"Train dates : {train_data.index.min()} --- {train_data.index.max()}  (n={len(train_data)})")
    print(f"Test dates  : {test_data.index.min()} --- {test_data.index.max()}  (n={len(test_data)})")

    if visualize:
        fig, ax = plt.subplots(figsize=(9, 4))
        train_data["y"].plot(ax=ax, label='train')
        test_data["y"].plot(ax=ax, label='test')
        ax.legend()
        plt.show()

    return train_data, test_data


def preprocess(df, horizon=2021, for_visuals=False):
    # Drop all rows with entry "Summe"
    df = df.drop(df[df.MONAT == "Summe"].index)

    # Drop all rows that are not from type "insgesamt"
    df = df.drop(df[df.AUSPRÄGUNG != "insgesamt"].index)

    # Drop all rows with year > 2021
    df = df.drop(df[df.JAHR > horizon].index)

    # Rename columns
    df = df.rename(columns={'MONAT': 'ds'})
    df = df.rename(columns={'WERT': 'y'})

    df['ds'] = pd.to_datetime(df['ds'], format='%Y%m')

    # Drop columns
    if for_visuals:
        df = df.drop(["AUSPRÄGUNG", "JAHR"], axis=1)
        df = df.rename(columns={'MONATSZAHL': 'category'})

    else:
        # Drop all rows that do not belong to category "Alkoholunfälle"
        df = df.drop(df[df.MONATSZAHL != "Alkoholunfälle"].index)

        df = df.drop(["MONATSZAHL", "AUSPRÄGUNG", "JAHR"], axis=1)

        # Change type of "date" column to datetime
        df['ds'] = pd.to_datetime(df['ds'], format='%Y%m')

        df = sort_data(df)

        # Check for missing values
        print(f'Number of rows with missing values: {df.isnull().any(axis=1).mean()}')

    return df


def sort_data(df):
    df = df.set_index('ds', drop=False)
    df = df.asfreq('MS')
    df = df.sort_index()
    return df


def save_model(filename, model):
    filepath = os.path.join("saved_models", filename + ".pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved at path: {filepath}")


def load_model(filename):
    filepath = os.path.join("saved_models", filename + ".pkl")
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"Load Model from path: {filepath}")
    return model


def return_pred_value(y_pred, month, year):
    # Predict single value given a specific date (year, month)
    date = year + '-' + month + '-01'
    predicted_value = int(y_pred.loc[date])

    print(f"\nPredicted number of accidents in {year}-{month}: {predicted_value}\n")

    return predicted_value


def calc_mse(y, y_pred, steps=24):
    error_mse = mean_squared_error(
        y_true=y,
        y_pred=y_pred
    )
    print(f"Test error (MSE) for prediction horizon of {steps} steps: {error_mse}")

