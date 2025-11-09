#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py - Workout Type Recommendation Flask API

This script creates a Flask web service that serves the trained model
for making workout type predictions.

Usage:
    python predict.py
    
Then access:
    http://localhost:5000/
"""

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = 'models/final_model.pkl'
HOST = '0.0.0.0'  # Allow external connections (important for Docker)
PORT = 5000

# ============================================
# LOAD MODEL
# ============================================

print("="*60)
print("LOADING MODEL")
print("="*60)

with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
label_encoder = model_package['label_encoder']
scaler = model_package['scaler']
feature_names = model_package['feature_names']
classes = model_package['classes']

print(f"✅ Model loaded successfully!")
print(f"   Model type: {model_package['model_type']}")
print(f"   Classes: {classes}")
print(f"   Test accuracy: {model_package['test_accuracy']:.2%}")
print(f"   Features: {len(feature_names)}")

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)

# ============================================
# HELPER FUNCTIONS
# ============================================

def prepare_features(input_data):
    """
    Prepare input data to match the training feature format.
    
    Args:
        input_data: Dictionary with user input
        
    Returns:
        DataFrame with properly formatted and scaled features
    """
    # Extract basic inputs
    age = input_data.get('age')
    gender = input_data.get('gender')
    weight = input_data.get('weight_kg')
    height = input_data.get('height_m')
    max_bpm = input_data.get('max_bpm')
    avg_bpm = input_data.get('avg_bpm')
    resting_bpm = input_data.get('resting_bpm')
    session_duration = input_data.get('session_duration_hours')
    calories_burned = input_data.get('calories_burned')
    fat_percentage = input_data.get('fat_percentage')
    water_intake = input_data.get('water_intake_liters')
    workout_frequency = input_data.get('workout_frequency_days_week')
    experience_level = input_data.get('experience_level')  # 'Beginner', 'Intermediate', 'Expert'
    
    # Calculate BMI
    bmi = weight / (height ** 2)
    
    # Engineered features
    bmi_category = 'Normal'
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif bmi >= 25 and bmi < 30:
        bmi_category = 'Overweight'
    elif bmi >= 30:
        bmi_category = 'Obese'
    
    age_group = '18-25'
    if age > 25 and age <= 35:
        age_group = '26-35'
    elif age > 35 and age <= 45:
        age_group = '36-45'
    elif age > 45:
        age_group = '46+'
    
    workout_intensity = (avg_bpm / max_bpm) * 100
    heart_rate_reserve = max_bpm - resting_bpm
    calories_per_hour = calories_burned / session_duration
    
    # Fitness score (same formula as training)
    fitness_score = (
        (workout_frequency / 7 * 30) +
        (water_intake / 4 * 20) +
        ((220 - age - resting_bpm) / 100 * 30) +
        ((100 - fat_percentage) / 100 * 20)
    )
    
    # Create feature dictionary matching training format
    features = {
        'Age': age,
        'Weight_kg': weight,
        'Height_m': height,
        'BMI': bmi,
        'Max_BPM': max_bpm,
        'Avg_BPM': avg_bpm,
        'Resting_BPM': resting_bpm,
        'Session_Duration_hours': session_duration,
        'Calories_Burned': calories_burned,
        'Fat_Percentage': fat_percentage,
        'Water_Intake_liters': water_intake,
        'Workout_Frequency_days_week': workout_frequency,
        'Workout_Intensity': workout_intensity,
        'Heart_Rate_Reserve': heart_rate_reserve,
        'Calories_per_Hour': calories_per_hour,
        'Fitness_Score': fitness_score
    }
    
    # Create DataFrame with all features initialized to 0
    df = pd.DataFrame([{feat: 0 for feat in feature_names}])
    
    # Fill in the numerical features
    for key, value in features.items():
        if key in df.columns:
            df[key] = value
    
    # Handle categorical features (one-hot encoded)
    # Gender
    if gender == 'Male':
        if 'Gender_Male' in df.columns:
            df['Gender_Male'] = 1
    # Note: Gender_Female would be 0 (dropped in one-hot encoding with drop_first=True)
    
    # Experience Level
    if experience_level == 'Intermediate':
        if 'Experience_Level_Intermediate' in df.columns:
            df['Experience_Level_Intermediate'] = 1
    elif experience_level == 'Expert':
        if 'Experience_Level_Expert' in df.columns:
            df['Experience_Level_Expert'] = 1
    # Beginner is baseline (all 0s)
    
    # BMI Category
    if bmi_category == 'Overweight':
        if 'BMI_Category_Overweight' in df.columns:
            df['BMI_Category_Overweight'] = 1
    elif bmi_category == 'Obese':
        if 'BMI_Category_Obese' in df.columns:
            df['BMI_Category_Obese'] = 1
    elif bmi_category == 'Underweight':
        if 'BMI_Category_Underweight' in df.columns:
            df['BMI_Category_Underweight'] = 1
    # Normal is baseline
    
    # Age Group
    if age_group == '26-35':
        if 'Age_Group_26-35' in df.columns:
            df['Age_Group_26-35'] = 1
    elif age_group == '36-45':
        if 'Age_Group_36-45' in df.columns:
            df['Age_Group_36-45'] = 1
    elif age_group == '46+':
        if 'Age_Group_46+' in df.columns:
            df['Age_Group_46+'] = 1
    # 18-25 is baseline
    
    return df

def make_prediction(features_df):
    """Make prediction using the loaded model."""
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction_encoded = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    # Decode prediction
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities for all classes
    probabilities = {
        classes[i]: float(prediction_proba[i])
        for i in range(len(classes))
    }
    
    return prediction, probabilities

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Home page with API documentation and test form."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workout Type Recommendation API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }
            pre {
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .status {
                padding: 10px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                margin: 20px 0;
            }
            .endpoint {
                background-color: #e7f3ff;
                padding: 15px;
                border-left: 4px solid #2196F3;
                margin: 10px 0;
            }
            input, select {
                width: 100%;
                padding: 8px;
                margin: 5px 0;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                background-color: #fff3cd;
                border-radius: 5px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏋️ Workout Type Recommendation API</h1>
            
            <div class="status">
                ✅ <strong>Status:</strong> API is running!<br>
                📊 <strong>Model:</strong> XGBoost Classifier<br>
                🎯 <strong>Classes:</strong> """ + ", ".join(classes) + """
            </div>
            
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <strong>GET /</strong> - This page
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
            </div>
            
            <div class="endpoint">
                <strong>POST /predict</strong> - Make a prediction<br>
                <strong>Content-Type:</strong> application/json
            </div>
            
            <h2>🧪 Test the API</h2>
            
            <form id="predictForm">
                <label>Age:</label>
                <input type="number" id="age" value="25" required>
                
                <label>Gender:</label>
                <select id="gender">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
                
                <label>Weight (kg):</label>
                <input type="number" step="0.1" id="weight" value="70" required>
                
                <label>Height (m):</label>
                <input type="number" step="0.01" id="height" value="1.75" required>
                
                <label>Max BPM:</label>
                <input type="number" id="max_bpm" value="180" required>
                
                <label>Average BPM:</label>
                <input type="number" id="avg_bpm" value="140" required>
                
                <label>Resting BPM:</label>
                <input type="number" id="resting_bpm" value="70" required>
                
                <label>Session Duration (hours):</label>
                <input type="number" step="0.1" id="session_duration" value="1.0" required>
                
                <label>Calories Burned:</label>
                <input type="number" id="calories" value="400" required>
                
                <label>Fat Percentage:</label>
                <input type="number" step="0.1" id="fat" value="20" required>
                
                <label>Water Intake (liters):</label>
                <input type="number" step="0.1" id="water" value="2.5" required>
                
                <label>Workout Frequency (days/week):</label>
                <input type="number" id="frequency" value="4" required>
                
                <label>Experience Level:</label>
                <select id="experience">
                    <option value="Beginner">Beginner</option>
                    <option value="Intermediate">Intermediate</option>
                    <option value="Expert">Expert</option>
                </select>
                
                <button type="submit">Get Recommendation</button>
            </form>
            
            <div id="result"></div>
            
            <h2>📝 Example cURL Request</h2>
            <pre>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 25,
    "gender": "Male",
    "weight_kg": 70,
    "height_m": 1.75,
    "max_bpm": 180,
    "avg_bpm": 140,
    "resting_bpm": 70,
    "session_duration_hours": 1.0,
    "calories_burned": 400,
    "fat_percentage": 20,
    "water_intake_liters": 2.5,
    "workout_frequency_days_week": 4,
    "experience_level": "Intermediate"
  }'
            </pre>
        </div>
        
        <script>
            document.getElementById('predictForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const data = {
                    age: parseInt(document.getElementById('age').value),
                    gender: document.getElementById('gender').value,
                    weight_kg: parseFloat(document.getElementById('weight').value),
                    height_m: parseFloat(document.getElementById('height').value),
                    max_bpm: parseInt(document.getElementById('max_bpm').value),
                    avg_bpm: parseInt(document.getElementById('avg_bpm').value),
                    resting_bpm: parseInt(document.getElementById('resting_bpm').value),
                    session_duration_hours: parseFloat(document.getElementById('session_duration').value),
                    calories_burned: parseFloat(document.getElementById('calories').value),
                    fat_percentage: parseFloat(document.getElementById('fat').value),
                    water_intake_liters: parseFloat(document.getElementById('water').value),
                    workout_frequency_days_week: parseInt(document.getElementById('frequency').value),
                    experience_level: document.getElementById('experience').value
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `
                        <h3>🎯 Recommendation: ${result.workout_type}</h3>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <h4>All Probabilities:</h4>
                        <ul>
                            ${Object.entries(result.probabilities).map(([k, v]) => 
                                `<li>${k}: ${(v * 100).toFixed(1)}%</li>`
                            ).join('')}
                        </ul>
                    `;
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': model_package['model_type'],
        'classes': classes
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects JSON with user features, returns workout type prediction.
    """
    try:
        # Get input data
        input_data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'age', 'gender', 'weight_kg', 'height_m',
            'max_bpm', 'avg_bpm', 'resting_bpm',
            'session_duration_hours', 'calories_burned',
            'fat_percentage', 'water_intake_liters',
            'workout_frequency_days_week', 'experience_level'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Prepare features
        features_df = prepare_features(input_data)
        
        # Make prediction
        prediction, probabilities = make_prediction(features_df)
        
        # Return result
        return jsonify({
            'workout_type': prediction,
            'confidence': float(max(probabilities.values())),
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER")
    print("="*60)
    print(f"Server running at: http://localhost:{PORT}")
    print("Press CTRL+C to stop")
    print("="*60 + "\n")
    
    app.run(host=HOST, port=PORT, debug=True)