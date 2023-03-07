import pyaf.ForecastEngine as autof

from utils import calc_mse


def train_model(train_data):
    """Fit PyAF model for forecasting"""

    train_data.index.name = 'date'

    model = autof.cForecastEngine()
    model.train(iInputDS=train_data, iTime='ds', iSignal="y", iHorizon=24)

    return model


def make_prediction(model, train_data, test_data, steps=24, show_stats=False):
    """Makes prediction for 24 months time horizon based on trained model"""

    predictions = model.forecast(train_data, steps)

    t_len = len(train_data['y'])
    predictions = predictions.set_index('ds', drop=False)
    y_pred = predictions["y_Forecast"][t_len:t_len+steps]

    if show_stats:
        calc_mse(test_data['y'], y_pred)

    return y_pred
