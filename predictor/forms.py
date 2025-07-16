from django import forms

class PredictForm(forms.Form):
    age = forms.IntegerField()
    sex = forms.ChoiceField(choices=[('male', 'Male'), ('female', 'Female')])
    bmi = forms.FloatField()
    children = forms.IntegerField()
    
