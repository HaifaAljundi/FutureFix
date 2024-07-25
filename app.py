from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
from datetime import datetime, timedelta
import pickle
import pandas as pd
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Load the models
binary_model_filename = "my_model.pkl"
regression_model_filename = "regression_model.pkl"

with open(binary_model_filename, 'rb') as f_in:
    binary_model = pickle.load(f_in)

with open(regression_model_filename, 'rb') as f_in:
    regression_model = pickle.load(f_in)

def predict_binary(data):
    try:
        df = pd.DataFrame(data[0])
        input_data = df.values.reshape((1, 50, 3)) 
        y_pred_prob = binary_model.predict(input_data)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        
        return 'Need Maintenance' if y_pred[0] == 1 else 'No Need Maintenance'
    except Exception as e:
        return {'error': str(e)}

def predict_regression(data):
    try:
        df = pd.DataFrame(data[0])
        input_data = df.values.reshape((1, 50, 3))
        y_pred = regression_model.predict(input_data)
        
        days_remaining = float(y_pred[0])

        start_date = datetime(2024, 1, 1)
        maintenance_date = (start_date + timedelta(days=days_remaining)).strftime('%Y-%m-%d')
        
        return {'days_remaining': days_remaining, 'maintenance_date': maintenance_date}
    except Exception as e:
        return {'error': str(e)}

class Test(Resource):
    def get(self):
        return 'Welcome to Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if value:
                return {'Post Values': value}, 201
            return {"error": "Invalid format."}, 400
        except Exception as error:
            return {'error': str(error)}, 500

class GetPredictionOutput(Resource):
    def get(self):
        return {"error": "Invalid Method."}, 405

    def post(self):
        try:
            data = request.get_json()
            if not data or 'model_type' not in data or 'data' not in data:
                return {'error': 'Invalid request format.'}, 400
            
            model_type = data['model_type']
            
            if model_type == 'binary':
                predict_output = predict_binary(data['data'])
            elif model_type == 'regression':
                predict_output = predict_regression(data['data'])
            else:
                return {'error': 'Invalid model type specified.'}, 400
            
            return {'predict': predict_output}, 200
        except Exception as error:
            return {'error': str(error)}, 500

api.add_resource(Test, '/')
api.add_resource(GetPredictionOutput, '/getPredictionOutput')

if __name__ == '__main__':
    app.run(debug=True)
