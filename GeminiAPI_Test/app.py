from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_restful import Api, Resource
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests
import google.generativeai as genai

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={"response_mime_type": "application/json"}
)

# Load models
binary_model_filename = "my_model.pkl"
regression_model_filename = "regression_model.pkl"

with open(binary_model_filename, 'rb') as f_in:
    binary_model = pickle.load(f_in)

with open(regression_model_filename, 'rb') as f_in:
    regression_model = pickle.load(f_in)

def call_gemini_api(input_data):
    try:
        prompt = f"""
        Analyze the following input data and provide detailed insights:
        {input_data}

        Please include:
        1. Key trends or patterns
        2. Anomalies or outliers
        3. Potential correlations between variables
        4. Actionable recommendations based on the data

        Format the response as a structured JSON object.
        """
        response = model.generate_content(prompt)
        print(f"Response from Gemini API: {response.text}")
        return response.text
    except genai.types.GenerateContentError as e:
        print(f"Error generating content: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def extract_contextual_insights(gemini_response):
    try:
        insights = json.loads(gemini_response)
        return insights
    except json.JSONDecodeError:
        return 'No insights available'

def predict_binary(df):
    try:
        input_data = df.values.reshape((3733, 40, 3))
        gemini_response = call_gemini_api(input_data)
        
        if 'error' in gemini_response:
            return gemini_response['error']
        
        # Process the response as needed
        y_pred_prob = binary_model.predict(input_data)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        y_pred_value = y_pred.flatten()[0]
        insights = extract_contextual_insights(gemini_response)

        return {
            'prediction': 'Need Maintenance' if y_pred_value == 0 else 'No Need Maintenance',
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
        
        # Process the response as needed
        y_pred = regression_model.predict(input_data)
        days_remaining = max(0, round(float(y_pred.flatten()[0])))
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getPredictionOutput', methods=['POST'])
def get_prediction_output():
    global stored_response
    try:
        # Check if the 'file' key exists in the request files
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading.'}), 400

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Call your prediction functions
        binary_prediction = predict_binary(df)
        regression_prediction = predict_regression(df)

        # Ensure that the predictions are structured correctly
        if not binary_prediction or not regression_prediction:
            return jsonify({'error': 'Prediction failed, check the input data.'}), 500

        # Store the response in a structure
        stored_response = {
            'prediction_Prob': {
                'prediction': binary_prediction.get('prediction', 'No prediction available'),
                'insights': binary_prediction.get('insights', 'No insights available')
            },
            'prediction_day': {
                'days_remaining': regression_prediction.get('days_remaining', 'No data available'),
                'maintenance_date': regression_prediction.get('maintenance_date', 'No data available'),
                'insights': regression_prediction.get('insights', 'No detailed insights available')
            }
        }

        return jsonify(stored_response)

    except Exception as e:
        # Return an error message if any exception occurs
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)