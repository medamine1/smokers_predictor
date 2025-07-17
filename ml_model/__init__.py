from .utils import predict

test_person = {
    'age': 55,
    'height(cm)': 165,
    'weight(kg)': 60,
    'waist(cm)': 81.0,
    'eyesight(left)': 0.5,
    'eyesight(right)': 0.6,
    'hearing(left)': 1,
    'hearing(right)': 1,
    'systolic': 135,
    'relaxation': 87,
    'fasting blood sugar': 94,
    'Cholesterol': 172,
    'triglyceride': 300,
    'HDL': 40,
    'LDL': 75,
    'hemoglobin': 16.5,
    'Urine protein': 1,
    'serum creatinine': 1.0,
    'AST': 22,
    'ALT': 25,
    'Gtp': 27,
    'dental caries': 0
}

prediction, proba = predict(test_person)
print(f"Prediction: {'Smoker' if prediction == 1 else 'Non-smoker'}")
print(f"Probability: {proba:.2%}")
