import numpy as np

labels = np.load("labels.npy")
print(f"Total signs: {len(labels)}")
print("Signs the model can recognize:")
for i, label in enumerate(labels):
    print(f"{i}: {label}")
