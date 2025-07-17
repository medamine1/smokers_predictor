from utils import predict

samples = [

    {
        'age': 45,
        'height_cm': 175,
        'weight_kg': 100,
        'waist_cm': 101.0,
        'eyesight_left': 0.9,
        'eyesight_right': 0.8,
        'hearing_left': 1,
        'hearing_right': 1,
        'systolic': 110,
        'relaxation': 78,
        'fasting_blood_sugar': 102,
        'Cholesterol': 195,
        'triglyceride': 289,
        'HDL': 34,
        'LDL': 103,
        'hemoglobin': 17.3,
        'Urine_protein': 1,
        'serum_creatinine': 1.1,
        'AST': 49,
        'ALT': 79,
        'Gtp': 68,
        'dental_caries': 0
    }
]


for test_person in samples:

 prediction, proba = predict(test_person)
 print(f"Prediction: {'Smoker' if prediction == 1 else 'Non-smoker'}")
 print(f"confidence: {proba:.2%}")
