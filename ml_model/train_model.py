import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/ELYADRIMohammedAmine/Downloads/train.csv')
df = df.dropna()
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df)  # Scales all columns by default
scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
X = scaled_df.drop('smoking', axis=1)
y = scaled_df['smoking']
model = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy_score * 100:.2f}%")
Classification_repo = classification_report(y_test, y_pred)
print("Classification Report:\n", Classification_repo)


