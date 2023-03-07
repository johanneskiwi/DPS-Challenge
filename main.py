import AR_model
import pyaf_model

from utils import save_model, load_model, load_dataset, preprocess, create_train_test_set, return_pred_value
from data_visualization import plot_input_data, plot_predictions

dataset = r'data.csv'
MODEL_NAME = "AR_model"
TRAIN_MODEL = False
MAKE_PREDICTION = True

PLOT_INPUT_DATA = False
PLOT_PREDICTION = True

pred_year = "2021"
pred_month = "03"

if __name__ == "__main__":
    # Load dataset:
    data = load_dataset(dataset)

    # Visualize input data
    if PLOT_INPUT_DATA:
        plot_input_data(dataset)

    # Preprocess Dataframe
    data = preprocess(data, for_visuals=False)

    # Get Train/Test Data
    train_data, test_data = create_train_test_set(data)

    # Visualize input data

    # Train model
    if TRAIN_MODEL:
        if MODEL_NAME == "AR_model":
            model = AR_model.train_model(train_data)

        elif MODEL_NAME == "pyaf_model":
            model = pyaf_model.train_model(train_data)

        else:
            raise NotImplementedError

        # Save model as pickle file
        save_model(MODEL_NAME, model)

    if MAKE_PREDICTION:
        # Load model
        model = load_model(MODEL_NAME)

        if MODEL_NAME == "AR_model":
            # Make prediction for AR model
            y_pred = AR_model.make_prediction(model, test_data)

        elif MODEL_NAME == "pyaf_model":
            y_pred = pyaf_model.make_prediction(model, test_data, show_stats=True)

        else:
            raise NotImplementedError

        # Plot train data with prediction data
        if PLOT_PREDICTION:
            plot_predictions(train_data, test_data, y_pred, steps=24)

        # Forecast single value for specific date:
        app_output = return_pred_value(y_pred, pred_month, pred_year)
