import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils import load_data

# Load dataset
X_train, y_train, label_map = load_data("dataset/train")
X_test, y_test, _ = load_data("dataset/test")

# Save label names
label_names = list(label_map.keys())
np.save("labels.npy", label_names)

num_classes = len(label_names)
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/hand_model.h5")

print("✅ Model trained and saved successfully!")