import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")

label_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
reverse_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
data['label'] = data['label'].map(label_map)


fft_data = data.loc[:, 'fft_0_b':'fft_749_b']

scaler = StandardScaler()
fft_data_scaled = scaler.fit_transform(fft_data)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(fft_data_scaled)
data['Cluster'] = kmeans.labels_

pca = PCA(n_components=2)
fft_data_pca = pca.fit_transform(fft_data_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(fft_data_pca[:, 0], fft_data_pca[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Clusters K-Means reducidos con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()

conf_matrix = pd.crosstab(data['Cluster'], data['label'])

conf_matrix.columns = [reverse_map[c] for c in conf_matrix.columns]

print("\nMatriz de comparación (Clusters vs Emoción Real):")
print(conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Clusters vs Emoción Real')
plt.xlabel('Emoción Real')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

