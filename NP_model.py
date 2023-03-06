import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error

from utils import load_dataset, preprocess


def train_model(file, visualize=False):
    data = load_dataset(file)

    # Preprocess Dataframe
    data = preprocess(data, for_visuals=False)

    # Rename columns for NeuralProphet model
    data.rename(columns={'date': 'ds', 'value': 'y'}, inplace=True)

    print(data.head())

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

    # Fit Autoregression model for forecasting
    model = NeuralProphet()
    metrics = model.fit(train_data, freq='M', validation_df=test_data)

    return model, train_data, test_data, metrics


def make_prediction(model, train_data, test_data, steps=24):
    future = model.make_future_dataframe(train_data, periods=steps, n_historic_predictions=len(train_data))
    predictions = model.predict(future)

    fig, ax = plt.subplots(figsize=(9, 4))
    train_data['y'].plot(ax=ax, label='train')
    test_data['y'][:steps].plot(ax=ax, label='test')

    l = len(train_data['y'])

    predictions = predictions.set_index('ds', drop=False)
    y_pred = predictions["yhat1"][l:l+steps]

    print(y_pred)

    y_pred.plot(ax=ax, label='predictions')
    ax.legend()

    model.plot(predictions)
    plt.show()

    error_mse = mean_squared_error(
        y_true=test_data['y'],
        y_pred=y_pred
    )

    print(f"Test error (mse): {error_mse}")


def plot_training_stats(metrics):
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(metrics["MAE"], '-o', label="Training Loss")
    ax.plot(metrics["MAE_val"], '-r', label="Validation Loss")
    ax.legend(loc='center right', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Epoch", fontsize=28)
    ax.set_ylabel("Loss", fontsize=28)
    ax.set_title("Model Loss (MAE)", fontsize=28)


if __name__ == "__main__":
    # Train NP Model
    model, train_data, test_data, stats = train_model("data.csv")

    # Show performance
    plot_training_stats(stats)

    # Make prediction
    make_prediction(model, train_data, test_data)

