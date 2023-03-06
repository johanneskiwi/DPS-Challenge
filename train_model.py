import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster

from utils import load_dataset, preprocess


def train_model(file, visualize=False):
    data = load_dataset(file)

    # Preprocess Dataframe
    data = preprocess(data)

    print(data.head())

    # Training data only for JAHR < 2020
    train_data = data[data["date"] <= '2020-12-01']
    test_data = data[data["date"] > '2020-12-01']

    print(f"Train dates : {train_data.index.min()} --- {train_data.index.max()}  (n={len(train_data)})")
    print(f"Test dates  : {test_data.index.min()} --- {test_data.index.max()}  (n={len(test_data)})")

    if visualize:
        fig, ax = plt.subplots(figsize=(9, 4))
        train_data["value"].plot(ax=ax, label='train')
        test_data["value"].plot(ax=ax, label='test')
        ax.legend()
        plt.show()

    # Fit Autoregression model for forecasting
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(max_depth=7, n_estimators=100, random_state=123),
        lags=50
    )

    forecaster.fit(y=train_data['value'])

    return forecaster, train_data, test_data


def make_prediction(model, train_data, test_data):
    steps = 12
    predictions = model.predict(steps=steps)

    fig, ax = plt.subplots(figsize=(9, 4))
    train_data['value'].plot(ax=ax, label='train')
    test_data['value'].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend()
    plt.show()

    error_mse = mean_squared_error(
        y_true=test_data['value'],
        y_pred=predictions
    )

    print(f"Test error (mse): {error_mse}")


def hyperparameter_tuning(train_data, steps=12):
    # Hyperparameter Grid search
    # ==============================================================================
    model = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=42),
        lags=12
    )

    # Lags used as predictors
    lags_grid = [5, 25]

    # Regressor's hyperparameters
    param_grid = {'n_estimators': [100, 500],
                  'max_depth': [3, 5, 7, 10]}

    results_grid = grid_search_forecaster(
        forecaster=model,
        y=train_data['value'],
        param_grid=param_grid,
        lags_grid=lags_grid,
        steps=steps,
        refit=True,
        metric='mean_squared_error',
        initial_train_size=int(len(train_data) * 0.5),
        fixed_train_size=False,
        return_best=True,
        verbose=False
    )

    print(results_grid)


if __name__ == "__main__":

    forecaster, train_data, test_data = train_model("data.csv")

    #hyperparameter_tuning(train_data)
    make_prediction(forecaster, train_data, test_data)

