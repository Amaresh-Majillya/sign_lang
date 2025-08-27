from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
labels = np.load("labels.npy")

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save model + labels
os.makedirs("model", exist_ok=True)
model.save("model/sign_model.h5")
np.save("model/labels.npy", labels)

print("Training complete! Model saved in model/")
