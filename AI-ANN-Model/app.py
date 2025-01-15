import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras 
from keras import Sequential
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

file_path = 'heart.csv'  
heart_data = pd.read_csv(file_path)

if heart_data.isnull().sum().any():
    print("Missing values detected. Filling with column mean value.")
    heart_data.fillna(heart_data.mean(), inplace=True)

for column in heart_data.columns:
    if heart_data[column].dtype == 'object':  
        print(f"Encoding non-numeric column: {column}")
        encoder = LabelEncoder()
        heart_data[column] = encoder.fit_transform(heart_data[column])

X = heart_data.iloc[:, :-1] 
y = heart_data.iloc[:, -1] 

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_clusters = kmeans.fit_predict(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    keras.layers.Dense(32, input_dim=X_train.shape[1], activation='relu'),  
    keras.layers.Dropout(0.3),  
    keras.layers.Dense(16, activation='relu'),  
    keras.layers.Dropout(0.2),  
    keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=125, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()