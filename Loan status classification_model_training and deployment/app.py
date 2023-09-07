from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)
# loading the model
model = pickle.load(open('xg_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# function to preprocess our data from train models
def preprocessing_data(data):
    numerical_data = ['loan_amount', 'lender_count', 'repayment_term']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[numerical_data] = scaler.fit_transform(data[numerical_data])
    return data

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = request.form
        user_input = pd.DataFrame([data], columns=data.keys())
        preprocessed_input = preprocessing_data(user_input)

    # make predictions using the preprocessed data
        prediction = model.predict(preprocessed_input)
        predicted_values = np.argmax(prediction, axis=1)
        status_labels = {0: 'expired', 1: 'fundraising', 2: 'funded'}
        predicted_labels = [status_labels.get(pred) for pred in predicted_values]
        return render_template('index.html', prediction = predicted_labels[0])
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)