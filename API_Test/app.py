from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Load models
binary_model_filename = "my_model.pkl"
regression_model_filename = "regression_model.pkl"

with open(binary_model_filename, 'rb') as f_in:
    binary_model = pickle.load(f_in)

with open(regression_model_filename, 'rb') as f_in:
    regression_model = pickle.load(f_in)

def predict_binary(df):
    try:
        input_data = df.values.reshape((2240, 50, 3))
        y_pred_prob = binary_model.predict(input_data)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        return 'Need Maintenance' if y_pred[0] == 1 else 'No Need Maintenance'
    except Exception as e:
        return {'error': str(e)}

def predict_regression(df):
    try:
        input_data = df.values.reshape((2240, 50, 3))
        y_pred = regression_model.predict(input_data)
        days_remaining = int(round(float(y_pred[0])))
        start_date = datetime(2024, 1, 1)
        maintenance_date = (start_date + timedelta(days=days_remaining)).strftime('%Y-%m-%d')
        return {'days_remaining': days_remaining, 'maintenance_date': maintenance_date}
    except Exception as e:
        return {'error': str(e)}

class Test(Resource):
    def get(self):
        return jsonify({'message': 'Welcome to Test App API!'})

    def post(self):
        try:
            value = request.get_json()
            if value:
                return jsonify({'Post Values': value}), 201
            return jsonify({"error": "Invalid format."}), 400
        except Exception as error:
            return jsonify({'error': str(error)}), 500

class GetPredictionOutput(Resource):
    def post(self):
        global stored_response, stored_df
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part in the request.'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading.'}), 400

            df = pd.read_csv(file)
            stored_df = df

            binary_prediction = predict_binary(df)
            regression_prediction = predict_regression(df)

            stored_response = {
                'prediction_Prob': binary_prediction,
                'prediction_day': regression_prediction
            }

            return stored_response , 200
        except Exception as error:
            return jsonify({'error': str(error)}), 500
        
    def get(self):
        global stored_response
        if stored_response:

            return stored_response, 200
        else:
            return jsonify({"error": "No stored response available."}), 404
        

api.add_resource(Test, '/')
api.add_resource(GetPredictionOutput, '/getPredictionOutput')

if __name__ == '__main__':
    app.run(debug=True)
