import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Load models
models = {
    'Model_A': tf.keras.models.load_model('my_model.h5'),
    'Model_B': tf.keras.models.load_model('my_model-SGD.h5'),
    'Model_C': tf.keras.models.load_model('model-nasnet.h5'),
    'Model_D': tf.keras.models.load_model('densenet-model.h5')
}

# Create a simple test image with a gradient
test_img = np.zeros((1, 224, 224, 3))
for i in range(224):
    for j in range(224):
        test_img[0, i, j, :] = [i/224, j/224, (i+j)/(2*224)]

# Get predictions from each model
print("Model predictions on test gradient image:")
for name, model in models.items():
    pred = model.predict(test_img, verbose=0)
    print(f"{name}: {pred}")
    print(f"Predicted class: {np.argmax(pred)}")
    print(f"Confidence: {np.max(pred):.4f}")
    print()

# Compare model architectures
print("\nModel architecture comparison:")
for name, model in models.items():
    print(f"\nModel: {name}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Check weights of first few layers
    for i in range(min(3, len(model.layers))):
        if len(model.layers[i].get_weights()) > 0:
            weights = model.layers[i].get_weights()[0]
            print(f"Layer {i} weights shape: {weights.shape}")
            print(f"Layer {i} weights mean: {np.mean(weights):.6f}")
            print(f"Layer {i} weights std: {np.std(weights):.6f}")
