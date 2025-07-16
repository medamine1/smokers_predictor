from django import forms

class PredictForm(forms.Form):
    age = forms.IntegerField()
    height_cm = forms.FloatField(label='Height (cm)')
    weight_kg = forms.FloatField(label='Weight (kg)')
    waist_cm = forms.FloatField(label='Waist (cm)')
    eyesight_left = forms.FloatField(label='Eyesight (left)')
    eyesight_right = forms.FloatField(label='Eyesight (right)')
    hearing_left = forms.IntegerField(label='Hearing (left)')  # 0 or 1?
    hearing_right = forms.IntegerField(label='Hearing (right)')
    systolic = forms.FloatField()
    relaxation = forms.FloatField()
    fasting_blood_sugar = forms.FloatField()
    Cholesterol = forms.FloatField()
    triglyceride = forms.FloatField()
    HDL = forms.FloatField()
    LDL = forms.FloatField()
    hemoglobin = forms.FloatField()
    Urine_protein = forms.IntegerField()
    serum_creatinine = forms.FloatField()
    AST = forms.FloatField()
    ALT = forms.FloatField()
    Gtp = forms.FloatField()
    dental_caries = forms.IntegerField()  # 0 or 1
