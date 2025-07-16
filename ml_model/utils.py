import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
model = joblib.load(MODEL_PATH)

def predict(input_data):
  
    features = [
        input_data['age'],
        input_data['height(cm)'],
        input_data['weight(kg)'],
        input_data['waist(cm)'],
        input_data['eyesight(left)'],
        input_data['eyesight(right)'],
        input_data['hearing(left)'],
        input_data['hearing(right)'],
        input_data['systolic'],
        input_data['relaxation'],
        input_data['fasting blood sugar'],
        input_data['Cholesterol'],
        input_data['triglyceride'],
        input_data['HDL'],
        input_data['LDL'],
        input_data['hemoglobin'],
        input_data['Urine protein'],
        input_data['serum creatinine'],
        input_data['AST'],
        input_data['ALT'],
        input_data['Gtp'],
        input_data['dental caries']
    ]

    features_array = np.array([features])
    scaled_features = scaler.transform(features_array)

    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    return prediction, probability