from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

app1 = Flask(__name__)
# loading the model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
expected_columns = ['loan_amount', 'lender_count','expired', 'funded', 'fundraising', 'Agriculture','Arts','Clothing','Construction','Education','Food','Health','Housing','Manufacturing','Personal Use','Retail','Services','Transportation','Wholesale']

@app1.route('/')
def home():
    return render_template('index.html')

# function to preprocess our data from the user input
from sklearn.preprocessing import MinMaxScaler
# function to preprocess our data from train models
def preprocessing_data(data):

    # Convert the following numerical labels from interger to float
    float_array = data.select_dtypes(include = [float, int])

    # categorical features to be converted to One Hot Encoding
    categ = ['status', 'sector']
                                                                            
    # One Hot Encoding conversion
    data = pd.get_dummies(data, columns=categ, prefix='', prefix_sep='').astype(int)
    
    missing_cols = set(expected_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # reordering the columns to match the expected order from training
    data = data[expected_columns]
    # scale our data into range of 0 and 1 
    # scaler = MinMaxScaler(feature_range=(0, 1))
    data[float_array.columns] = scaler.transform(float_array)
    data = pd.DataFrame(data, columns=float_array.columns.tolist() + data.drop(columns=float_array.columns).columns.tolist())
    
    return data                  

@app1.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get individual input from the user
        loan_amount = float(request.form.get('loan_amount'))
        lender_count = float(request.form.get('lender_count'))
        status = request.form.get('status')
        sector = request.form.get('sector')

        # Create a dictionary with the user input
        user_input = {
            'loan_amount': [loan_amount],
            'lender_count': [lender_count],
            'status': [status],
            'sector': [sector]
        }
        # Convert the user input to a DataFrame
        user_input_df = pd.DataFrame(user_input)

        # Preprocess the user input
        preprocessed_input = preprocessing_data(user_input_df)

        # user_input_array = np.array(preprocessed_input)

        # Make predictions using the preprocessed data
        prediction = model.predict(preprocessed_input)

        return render_template('index.html', prediction=int(prediction[0]))

    else:
        return render_template('index.html')

if __name__ == '__main__':
    print("starting the flask app...")
    app1.run(debug=True)
