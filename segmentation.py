import cv2
import numpy as np
import matplotlib.pyplot as plt
import azClassifier as az

#Load weights
az.model.load_weights("az_ocr.weights.h5")

# Load and preprocess image
img = cv2.imread("Illustration84.png", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)


contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Center individual letters for predictions
def resize_and_center(img, size=28):
    h, w = img.shape
    scale = size / max(h, w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Create blank square
    canvas = np.zeros((size, size), dtype=np.uint8)

    # Center the character
    y_offset = (size - resized.shape[0]) // 2
    x_offset = (size - resized.shape[1]) // 2
    canvas[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    return canvas


# Segment and process letters for prediction
letters = []
for c in contours:
    if cv2.contourArea(c) > 50:
        x, y, w, h = cv2.boundingRect(c)
        pad = 3
        x = max(x - pad, 0)
        y = max(y - pad, 0)
        w = min(w + pad*2, thresh.shape[1] - x)
        h = min(h + pad*2, thresh.shape[0] - y)
        char_img = thresh[y:y+h, x:x+w]
        char_img = resize_and_center(char_img)
        letters.append(char_img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)


# Letter prediction
predictions = []
for char in letters:
    char = char.astype('float32') / 255.0
    # Reshape to match CNN input: (1, 28, 28, 1)
    char = np.expand_dims(char, axis=(0, -1))
    pred = az.model.predict(char, verbose=0)
    label = np.argmax(pred)
    predictions.append(chr(label + 65))

print("Predicted text:", "".join(predictions))

# Display results
plt.figure(figsize=(10, 3))
for i, letter in enumerate(letters):
    plt.subplot(1, len(letters), i+1)
    plt.imshow(letter.squeeze(), cmap="gray")
    plt.title(predictions[i])
    plt.axis("off")
plt.show()

