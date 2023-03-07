from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from utils import calc_mse


def train_model(train_data):
    """Fit Autoregression model for forecasting"""

    model = ForecasterAutoreg(
        regressor=RandomForestRegressor(max_depth=7, n_estimators=100, random_state=123),
        lags=30
    )

    model.fit(y=train_data['value'])

    return model


def make_prediction(model, test_data, steps=24, show_stats=False):
    """Make prediction for 24 months time horizon based on trained model"""

    y_pred = model.predict(steps=steps)

    if show_stats:
        calc_mse(test_data['y'], y_pred)

    return y_pred
