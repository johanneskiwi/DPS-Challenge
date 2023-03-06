import AR_model
import NP_model
import pyaf_model
from utils import save_model, load_model, load_dataset, preprocess, create_train_test_set

MODEL_NAME = "pyaf_model"
TRAIN_MODEL = False
MAKE_PREDICTION = True


if __name__ == "__main__":
    # Load dataset:
    data = load_dataset("data.csv")

    # Preprocess Dataframe
    data = preprocess(data, for_visuals=False)

    # Get Train/Test Data
    train_data, test_data = create_train_test_set(data)

    # Train model
    if TRAIN_MODEL:
        if MODEL_NAME == "AR_model":
            model = AR_model.train_model(train_data)

        elif MODEL_NAME == "NP_model":
            model = NP_model.train_model(train_data)

        elif MODEL_NAME == "pyaf_model":
            model = pyaf_model.train_model(train_data)

        else:
            raise NotImplementedError

        # Save AR model
        save_model(MODEL_NAME, model)

    if MAKE_PREDICTION:
        # Load model
        model = load_model(MODEL_NAME)

        if MODEL_NAME == "AR_model":
            # Make prediction for AR model
            AR_model.make_prediction(model, train_data, test_data)

        elif MODEL_NAME == "NP_model":
            # Make prediction for NP model
            # TODO // Does not work yet => some weird error occuring
            NP_model.make_prediction(model, train_data, test_data)

        elif MODEL_NAME == "pyaf_model":
            model = pyaf_model.make_prediction(model, train_data, test_data)

        else:
            raise NotImplementedError
