from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.loc[:, 'fft_0_b':'fft_749_b']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = [1, 3, 5]
accuracies = []

for k in k_values:
    print(f"\n===== KNN con k={k} =====")
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_knn)
    accuracies.append(acc)

    print(f"Test Accuracy (k={k}): {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_knn, target_names=le.classes_))

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d",
                cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"KNN (k={k}) - Matriz de Confusión")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(7, 5))
plt.plot(k_values, accuracies, marker='o', color='blue')
plt.title("Precisión del modelo KNN según valor de K")
plt.xlabel("Valor de K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
