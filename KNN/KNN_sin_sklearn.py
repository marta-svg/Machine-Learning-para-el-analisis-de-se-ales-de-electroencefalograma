import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time

data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")

le = LabelEncoder()
data['label'] = le.fit_transform(data['label']) 

X = data.loc[:, 'fft_0_b':'fft_749_b'].values
y = data['label'].values
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def knn_predict(X_train, y_train, x_test, k):
    distances = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
    nearest = np.argsort(distances)[:k]
    top_labels = y_train[nearest]
    return Counter(top_labels).most_common(1)[0][0]

k_values = [1, 3, 5]
accuracies = []



for k in k_values:
    start_k = time.time()

    y_pred = [knn_predict(X_train, y_train, x, k) for x in X_test]
    y_pred = np.array(y_pred, dtype=int) 
    
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

    num_classes = len(le.classes_)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_test, y_pred):
        conf_matrix[int(true), int(pred)] += 1 

    print(f"\n===== KNN con k={k} =====")
    print(f"Test Accuracy (k={k}): {accuracy:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"KNN (k={k}) - Matriz de Confusión")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    end_k = time.time()
    print(f"⏱ Tiempo de ejecución para k={k}: {end_k - start_k:.2f} segundos")

plt.figure(figsize=(7, 5))
plt.plot(k_values, accuracies, marker='o', color='blue')
plt.title("Precisión del modelo KNN según valor de K")
plt.xlabel("Valor de K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
