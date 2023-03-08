# Serve model as a flask application
import pickle
from flask import Flask, request, jsonify

model = None
train_data = None
app = Flask(__name__)


def load_model():
    global model

    with open(r'pickle_files/pyaf_model.pkl', 'rb') as f:
        model = pickle.load(f)


def load_train_data():
    global train_data

    # Train data is necessary for forecaster
    with open(r'pickle_files/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    predicted_value = None

    if request.method == 'POST':
        usr_inp = request.get_json()
        year = usr_inp["year"]
        month = usr_inp["month"]

        # Make prediction
        predictions = model.forecast(iInputDS=train_data, iHorizon=24)
        t_len = len(train_data['y'])
        predictions = predictions.set_index('ds', drop=False)
        y_pred = predictions["y_Forecast"][t_len:t_len + 24]

        # Get predicted value
        date = year + '-' + month + '-01'
        predicted_value = int(y_pred.loc[date])

    return jsonify(prediction=predicted_value)


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    load_train_data()   # load training data at the beginning once only
    app.run(host='0.0.0.0', port=5000)

#curl.exe -X POST -H "Content-type: application/json" -d "{""year"": ""2021"", ""month"": ""12""}" "localhost:5000/predict"
