import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np

# Load Datafile
file_path = "A_Z Handwritten Data.csv"
data_Labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


def dataPrep():
    # Read file into DataFrame
    df = pd.read_csv(file_path).astype('float32')
    df.rename(columns={'0': 'label'}, inplace=True)

    # Features
    x = df.drop('label', axis=1)
    # Labels
    y = df['label']

    # Split into sets
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    # Normalize data
    standard_scaler = MinMaxScaler()
    standard_scaler.fit(X_train)

    X_train = standard_scaler.transform(X_train)
    X_test = standard_scaler.transform(X_test)

    # Reshape for CNN input
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    return X_train, X_test, y_train, y_test


# Convolutional Neural Network Model definition
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(26, activation="softmax")

])

weights_path = "az_ocr.weights.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    X_train, X_test, y_train, y_test = dataPrep()
    # Training Loop
    model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=5,
                        batch_size=128, shuffle=True)
    model.save_weights("az_ocr.weights.h5")

# Evaluation
def test(X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Visualization of predictions
    N = 16
    plt.figure(figsize=(12, 10))
    for i in range(N):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='Greys')
        plt.title(
            f"Pred: {data_Labels[predicted_classes[i]]}\nTrue: {data_Labels[int(y_test.iloc[i])]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

