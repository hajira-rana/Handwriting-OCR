import azClassifier as az
from azClassifier import data_Labels
import matplotlib.pyplot as plt
import numpy as np

#Load and compile model (defined in azClassifier)
az.model.load_weights("az_ocr.weights.h5")


def test(X_test, y_test):
    test_loss, test_acc = az.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    predictions = az.model.predict(X_test)
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

X_train, X_test, y_train, y_test = az.dataPrep()
test(X_test, y_test)