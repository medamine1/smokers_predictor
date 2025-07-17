from utils import predict

samples = [

 {
    'age': 20,
    'height_cm': 170,
    'weight_kg': 70,
    'waist_cm': 81.0,
    'eyesight_left': 1.5,
    'eyesight_right': 1.5,
    'hearing_left': 1,
    'hearing_right': 1,
    'systolic': 125,
    'relaxation': 80,
    'fasting_blood_sugar': 94,
    'Cholesterol': 149,
    'triglyceride': 104,
    'HDL': 43,
    'LDL': 85,
    'hemoglobin': 16.9,
    'Urine_protein': 1,
    'serum_creatinine': 1.1,
    'AST': 24,
    'ALT': 31,
    'Gtp': 33,
    'dental_caries': 0
}
]


for test_person in samples:

 prediction, proba = predict(test_person)
 print(f"Prediction: {'Smoker' if prediction == 1 else 'Non-smoker'}")
 print(f"confidence: {proba:.2%}")
