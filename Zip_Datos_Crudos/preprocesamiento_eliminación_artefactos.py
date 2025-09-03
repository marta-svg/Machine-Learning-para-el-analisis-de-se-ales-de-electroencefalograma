import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eeg_channels = ["EEG C3-A1", "EEG O1-A1", "EEG C4-A1", "EEG O2-A1"]
data = pd.DataFrame(np.random.randn(1000, len(eeg_channels)) * 10, columns=eeg_channels)

data.iloc[100:110, 0] = 500  
data.iloc[200:210, 2] = -400  

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(data["EEG C3-A1"], label="EEG C3-A1 (Antes)", color='b')
plt.title("EEG con Picos Anómalos (Antes)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

for ch in eeg_channels:
    threshold = np.percentile(data[ch], 99) 
    data[ch] = np.where(data[ch] > threshold, np.nan, data[ch])  

data.fillna(data.mean(), inplace=True)

plt.subplot(1, 2, 2)
plt.plot(data["EEG C3-A1"], label="EEG C3-A1 (Después)", color='r')
plt.title("EEG sin Picos Anómalos (Después)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()

print(data.head())
