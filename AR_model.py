import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster


def train_model(train_data):
    # Fit Autoregression model for forecasting
    model = ForecasterAutoreg(
        regressor=RandomForestRegressor(max_depth=7, n_estimators=100, random_state=123),
        lags=30
    )

    model.fit(y=train_data['value'])

    return model


def make_prediction(model, train_data, test_data, steps=24):
    predictions = model.predict(steps=steps)

    fig, ax = plt.subplots(figsize=(9, 4))
    train_data['value'].plot(ax=ax, label='train')
    test_data['value'][:steps].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend()
    plt.show()

    error_mse = mean_squared_error(
        y_true=test_data['value'][:steps],
        y_pred=predictions
    )

    print(f"Test error (mse): {error_mse}")


def hyperparameter_tuning(train_data, steps=12):
    # Hyperparameter Grid search
    # ==============================================================================
    model = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=30
    )

    # Lags used as predictors
    lags_grid = [20, 50]

    # Regressor's hyperparameters
    param_grid = {'n_estimators': [100, 250, 500],
                  'max_depth': [7, 10, 15, 20, 40]}

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


