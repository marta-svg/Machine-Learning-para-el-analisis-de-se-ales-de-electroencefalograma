import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import time

data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")
print(data.info())

fft_data = data.loc[:,'fft_0_b':'fft_749_b']
print(fft_data)

fft_data.iloc[0,:].plot(figsize=(15,10))
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.loc[:, 'fft_0_b':'fft_749_b']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

start_time = time.time()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")

print("Resultados de la evaluación cruzada:")
print(f"Accuracy para cada fold: {cv_scores}")
print(f"Precisión media: {cv_scores.mean():.3f}")
print(f"Desviación estándar: {cv_scores.std():.3f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(13, 6))  
plot_tree(
    decision_tree=model,  
    feature_names=X.columns.tolist(), 
    class_names=le.classes_.tolist(), 
    filled=True,  
    impurity=False,  
    fontsize=7,  
)

plt.show()
