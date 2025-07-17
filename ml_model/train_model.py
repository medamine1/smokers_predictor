import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


df = pd.read_csv('C:/Users/ELYADRIMohammedAmine/Downloads/train.csv')
df = df.dropna()

X = df.drop(['id', 'smoking'], axis=1)
y = df['smoking']
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)  # Scales all columns by default
scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
model = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
confidence = y_proba[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc * 100:.2f}%")

classification_repo = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_repo)

results = pd.DataFrame({
    'prediction': y_pred,
    'confidence': confidence
})

print(results.head())

# Save model and scaler
joblib.dump(model, 'ml_model/model.pkl')
joblib.dump(scaler, 'ml_model/scaler.pkl')

print(confidence)
