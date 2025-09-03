import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow import keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt;
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from tensorflow import keras
import seaborn as sns
import time

data = pd.read_csv("C:/Users/6003376/Desktop/Marta/tfg/emotions.csv")
print(data.info())

fft_data = data.loc[:,'fft_0_b':'fft_749_b']
print(fft_data)
fft_data.iloc[0,:].plot(figsize=(15,10))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

y = data.pop('label')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)
X_train = np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
X_test = np.array(X_test).reshape((X_test.shape[0],X_test.shape[1],1))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

inputs = tf.keras.Input(shape=(X_train.shape[1],1))
gru = tf.keras.layers.GRU(256, return_sequences=True)(inputs)
flat = Flatten()(gru)
outputs = Dense(3, activation='softmax')(flat)
model = tf.keras.Model(inputs, outputs)
model.summary()

tf.keras.utils.plot_model(model)
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

def train_model(model,x_train, y_train,x_test,y_test, save_to, epoch = 1):
        opt_adam = keras.optimizers.Adam(learning_rate=0.001)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(save_to + '_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))
        
        model.compile(optimizer=opt_adam,
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])
        
        history = model.fit(x_train,y_train,
                        batch_size=32,
                        epochs=epoch,
                        validation_data=(x_test,y_test),
                        callbacks=[es,mc,lr_schedule])
        
        saved_model = load_model(save_to + '_best_model.h5')
        return model,history

start_time = time.time()
model,history = train_model(model, X_train, y_train,X_test, y_test, save_to= './', epoch = 3) 
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo total de ejecución: {execution_time:.2f} segundos")

print("Test Accuracy: {:.3f}%".format(model_acc * 100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))
y_test = y_test.idxmax(axis=1)

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)
