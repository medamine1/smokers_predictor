import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score



df = pd.read_csv('C:/Users/ELYADRIMohammedAmine/Downloads/train.csv')
df = df.dropna()

X = df.drop(['id', 'smoking'], axis=1)
y = df['smoking']
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)  # Scales all columns by default
scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
model = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],       # Regularization strength (inverse)
    'solver': ['liblinear', 'lbfgs'],    # solvers that work with LogisticRegression
    'class_weight': [None, 'balanced']   # Handle class imbalance
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
print("Test accuracy with best model:", test_accuracy)
y_pred = best_model.predict(X_test)


classification_repo = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_repo)


joblib.dump(best_model, 'ml_model/model.pkl')
joblib.dump(scaler, 'ml_model/scaler.pkl')