from train_AR_model import train_model, make_prediction

if __name__ == "__main__":
    # Train AR model
    forecaster, train_data, test_data = train_model("data.csv")

    # Make prediction for AR model
    make_prediction(forecaster, train_data, test_data)
