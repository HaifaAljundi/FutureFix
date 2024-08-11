from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Define the API endpoint and headers
gemini_api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GOOGLE_API_KEY}"
headers = {
    'Content-Type': 'application/json',
}

# Load models
binary_model_filename = "my_model.pkl"
regression_model_filename = "regression_model.pkl"

with open(binary_model_filename, 'rb') as f_in:
    binary_model = pickle.load(f_in)

with open(regression_model_filename, 'rb') as f_in:
    regression_model = pickle.load(f_in)

def call_gemini_api(input_data):
    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Some prediction request based on input data"}
                    ]
                }
            ]
        }
        response = requests.post(gemini_api_endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()  
        else:
            return {'error': f"Gemini API request failed with status code {response.status_code}: {response.text}"}
    except Exception as e:
        return {'error': str(e)}

def extract_contextual_insights(gemini_response):
    insights = gemini_response.get('insights', 'No insights available')
    return insights

def predict_binary(df):
    try:
        input_data = df.values.reshape((3733, 40, 3))
        gemini_response = call_gemini_api(input_data)
        
        if 'error' in gemini_response:
            return gemini_response['error']
        
        y_pred_prob = gemini_response.get('predictions', [0])[0]
        y_pred = int(bool(y_pred_prob))
        insights = extract_contextual_insights(gemini_response)

        return {
            'prediction': 'Need Maintenance' if y_pred == 1 else 'No Need Maintenance',
            'insights': insights
        }
    except Exception as e:
        return {'error': str(e)}

def predict_regression(df):
    try:
        input_data = df.values.reshape((3733, 40, 3))
        gemini_response = call_gemini_api(input_data)
        
        if 'error' in gemini_response:
            return gemini_response['error']
        
        y_pred = gemini_response.get('predictions', [0])[0]
        days_remaining = max(0, round(float(y_pred)))
        start_date = datetime(2024, 8, 2)
        maintenance_date = (start_date + timedelta(days=days_remaining)).strftime('%Y-%m-%d')
        insights = extract_contextual_insights(gemini_response)
        
        return {
            'days_remaining': days_remaining,
            'maintenance_date': maintenance_date,
            'insights': insights
        }
    except Exception as e:
        return {'error': str(e)}

class Test(Resource):
    def get(self):
        return jsonify({'message': 'Welcome to Test FutureFix App!'})

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
        global stored_response
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part in the request.'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading.'}), 400

            df = pd.read_csv(file)

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
