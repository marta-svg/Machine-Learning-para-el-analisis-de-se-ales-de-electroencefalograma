import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar los datos
data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")

# Ver las primeras filas del dataset
print(data.head())

# Seleccionar solo las columnas de FFT
fft_data = data.loc[:, 'fft_0_b':'fft_749_b']

# Normalizar los datos
scaler = StandardScaler()
fft_data_scaled = scaler.fit_transform(fft_data)

# Verifica los primeros datos escalados
print(fft_data_scaled[:5])

# Encontrar el valor óptimo de K utilizando el método del codo
inertia = []  # Inercia (suma de las distancias cuadradas de los puntos a su centroide)

for k in range(1, 11):  # Evaluar para k de 1 a 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(fft_data_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', color='b')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.show()

# Establecer el número de clusters
k = 3  # Ajusta este valor según lo que obtuviste del método del codo

# Crear el modelo de K-Means
kmeans = KMeans(n_clusters=k, random_state=42)

# Ajustar el modelo a los datos
kmeans.fit(fft_data_scaled)

# Obtener las etiquetas de los clusters
labels = kmeans.labels_

# Añadir las etiquetas al dataframe original
data['Cluster'] = labels

# Ver las primeras filas con las etiquetas de clusters
print(data.head())

# Reducir la dimensionalidad a 2D usando PCA
pca = PCA(n_components=2)
fft_data_pca = pca.fit_transform(fft_data_scaled)

# Crear un gráfico de dispersión para visualizar los clusters
plt.figure(figsize=(8, 6))
plt.scatter(fft_data_pca[:, 0], fft_data_pca[:, 1], c=labels, cmap='viridis')
plt.title('Clusters de K-Means (PCA reducido)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Comparar los clusters con las etiquetas reales (si tienes etiquetas de emociones)
# Si las etiquetas de emociones están en 'label' y quieres comparar:
print(pd.crosstab(data['Cluster'], data['label']))
