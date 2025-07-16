from django.shortcuts import render
from .forms import PredictForm


def predict_view(request):
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            # Here you would typically process the form data and make predictions
            return render(request, 'predictor/result.html', {'form': form})
    else:
        form = PredictForm()

        return render(request, 'predictor/index.html', {'form': form})
