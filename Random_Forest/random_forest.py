import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.loc[:, 'fft_0_b':'fft_749_b']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
end_time = time.time()
execution_time = end_time - start_time

accuracy = accuracy_score(y_test, y_pred)
print(f"\n===== Random Forest =====")
print(f"Test Accuracy: {accuracy:.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest - Matriz de Confusión")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

