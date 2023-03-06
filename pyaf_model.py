import pyaf.ForecastEngine as autof
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt


def train_model(train_data):
    # Fit Autoregression model for forecasting
    train_data.index.name = 'date'

    model = autof.cForecastEngine()
    model.train(iInputDS=train_data, iTime='ds', iSignal="y", iHorizon=24)

    return model


def make_prediction(model, train_data, test_data, steps=24):
    predictions = model.forecast(train_data, steps)

    fig, ax = plt.subplots(figsize=(9, 4))
    train_data['y'].plot(ax=ax, label='train')
    test_data['y'][:steps].plot(ax=ax, label='test')

    l = len(train_data['y'])

    predictions = predictions.set_index('ds', drop=False)
    y_pred = predictions["y_Forecast"][l:l+steps]

    y_pred.plot(ax=ax, label='predictions')

    ax.legend()
    plt.show()

    error_mse = mean_squared_error(
        y_true=test_data['y'],
        y_pred=y_pred
    )

    print(f"Test error (mse): {error_mse}")


