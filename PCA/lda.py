import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Cargar los datos
data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")

# Mapear etiquetas a valores numéricos
label_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
data['label_num'] = data['label'].map(label_map)

# Seleccionar columnas FFT
fft_data = data.loc[:, 'fft_0_b':'fft_749_b']

# Normalizar los datos
scaler = StandardScaler()
fft_data_scaled = scaler.fit_transform(fft_data)

# Aplicar LDA
lda = LinearDiscriminantAnalysis(n_components=2)
fft_data_lda = lda.fit_transform(fft_data_scaled, data['label_num'])

# Visualización en 2D
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    fft_data_lda[:, 0], fft_data_lda[:, 1],
    c=data['label_num'],
    cmap='viridis', alpha=0.7
)

# Leyenda
from matplotlib.lines import Line2D
legend_labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
legend_colors = [plt.cm.viridis(i / 2) for i in range(3)]
legend_handles = [Line2D([0], [0], marker='o', color='w', label=label,
                         markerfacecolor=color, markersize=10)
                  for label, color in zip(legend_labels, legend_colors)]

plt.legend(handles=legend_handles, title='Emoción')
plt.title('Visualización LDA - Componentes Discriminantes 1 y 2')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.grid(True)
plt.tight_layout()
plt.show()
