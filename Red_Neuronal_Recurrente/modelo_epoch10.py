import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd  # Solo si `y_test` es un DataFrame con one-hot encoding

# Cargar el modelo entrenado
model = load_model("C:/Users/6003376/_best_model.h5") 

# Cargar los datos de prueba (esto depende de tu dataset, ajusta si es necesario)
# Aquí deberías cargar tus datos de prueba X_test y y_test correctamente.
# Por ejemplo, si usaste algún preprocesamiento en el entrenamiento, aplícalo aquí también.
X_test = np.load("C:/Users/6003376/Desktop/Marta/tfg/red neuronal recurrente/X_test.npy")  # Ejemplo: cargar datos desde un archivo
y_test = np.load("y_test.npy")  # Asegúrate de que este formato coincida con lo que usaste en el entrenamiento

# Evaluar precisión en el conjunto de prueba
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

# Hacer predicciones
y_pred_prob = model.predict(X_test)  # Probabilidades de cada clase
y_pred = np.argmax(y_pred_prob, axis=1)  # Convertir a etiquetas predichas

# Si `y_test` está en formato one-hot encoding, convertirlo a etiquetas
if isinstance(y_test, pd.DataFrame):  
    y_test_labels = y_test.idxmax(axis=1)  
else:
    y_test_labels = np.argmax(y_test, axis=1)

# Generar matriz de confusión
cm = confusion_matrix(y_test_labels, y_pred)
clr = classification_report(y_test_labels, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Imprimir reporte de clasificación
print("Classification Report:\n----------------------\n", clr)
