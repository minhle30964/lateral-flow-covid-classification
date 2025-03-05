import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set fixed random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Noise functions
def add_gaussian_noise(images, mean=0, std=0.1):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1)

def add_shadows(images, alpha=0.5):
    # Get the shape of the images
    batch_size, height, width, channels = images.shape
    
    # Create shadow with proper dimensions (batch_size, height, width)
    shadow = np.random.random((batch_size, height, width)) * alpha
    
    # Expand to match image dimensions (batch_size, height, width, channels)
    shadow = np.expand_dims(shadow, axis=-1)
    shadow = np.repeat(shadow, channels, axis=-1)
    
    return np.clip(images * (1 - shadow), 0, 1)

# Fixed apply_difficulty function
def apply_difficulty(data, noise_std=0, shadow_alpha=0, rotation_range=0, brightness_range=None):
    # Apply gaussian noise
    modified_data = data.copy()
    if noise_std > 0:
        print(f"Adding Gaussian noise with std={noise_std}")
        modified_data = add_gaussian_noise(modified_data, std=noise_std)
        print(f"After Gaussian noise - Mean: {np.mean(modified_data):.4f}, Std: {np.std(modified_data):.4f}")
    
    # Apply shadows
    if shadow_alpha > 0:
        print(f"Adding shadows with alpha={shadow_alpha}")
        modified_data = add_shadows(modified_data, alpha=shadow_alpha)
        print(f"After shadows - Mean: {np.mean(modified_data):.4f}, Std: {np.std(modified_data):.4f}")
    
    # Apply rotation and brightness changes
    if rotation_range > 0 or brightness_range is not None:
        print(f"Applying rotation={rotation_range}, brightness={brightness_range}")
        datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            brightness_range=brightness_range,
            fill_mode='nearest'
        )
        
        augmented = []
        for img in modified_data:
            img = img.reshape((1,) + img.shape)
            aug_img = next(datagen.flow(img, batch_size=1, seed=42))[0]
            augmented.append(aug_img)
        
        modified_data = np.array(augmented)
        print(f"After augmentation - Mean: {np.mean(modified_data):.4f}, Std: {np.std(modified_data):.4f}")
    
    return modified_data

# Load models
models = {
    'Model_A': tf.keras.models.load_model('my_model.h5'),
    'Model_B': tf.keras.models.load_model('my_model-SGD.h5'),
    'Model_C': tf.keras.models.load_model('model-nasnet.h5'),
    'Model_D': tf.keras.models.load_model('densenet-model.h5')
}

# Create a test dataset with 10 images
test_images = np.zeros((10, 224, 224, 3))
for i in range(10):
    # Create a different pattern for each image
    for x in range(224):
        for y in range(224):
            test_images[i, x, y, 0] = (x + i*10) / 224
            test_images[i, x, y, 1] = (y + i*5) / 224
            test_images[i, x, y, 2] = ((x+y) + i*15) / (2*224)

# Assign labels (5 of each class)
test_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Test different noise levels
difficulties = {
    'Clean': {'noise_std': 0, 'shadow_alpha': 0, 'rotation_range': 0, 'brightness_range': None},
    'Low Noise': {'noise_std': 0.1, 'shadow_alpha': 0.2, 'rotation_range': 10, 'brightness_range': [0.8, 1.2]},
    'Medium Noise': {'noise_std': 0.2, 'shadow_alpha': 0.4, 'rotation_range': 20, 'brightness_range': [0.6, 1.4]},
    'High Noise': {'noise_std': 0.3, 'shadow_alpha': 0.6, 'rotation_range': 30, 'brightness_range': [0.4, 1.6]},
    'Extreme Noise': {'noise_std': 0.5, 'shadow_alpha': 0.8, 'rotation_range': 45, 'brightness_range': [0.2, 1.8]}
}

# Visualize the effect of noise
plt.figure(figsize=(15, 10))
for i, (name, params) in enumerate(difficulties.items()):
    print(f"\nApplying {name} difficulty to test image")
    # Apply noise to the first image
    noisy_images = apply_difficulty(test_images[:1], **params)
    plt.subplot(1, 5, i+1)
    plt.imshow(noisy_images[0])
    plt.title(name)
plt.savefig('fixed_noise_levels.png')
plt.close()

# Test model predictions at each noise level
print("\n\nModel predictions at different noise levels:")
for name, params in difficulties.items():
    print(f"\n{name}:")
    noisy_images = apply_difficulty(test_images, **params)
    
    # Check image statistics
    print(f"Final image stats - Mean: {np.mean(noisy_images):.4f}, Std: {np.std(noisy_images):.4f}")
    
    all_preds = {}
    for model_name, model in models.items():
        preds = model.predict(noisy_images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        all_preds[model_name] = pred_classes
        
        # Calculate accuracy
        accuracy = np.mean(pred_classes == test_labels)
        print(f"{model_name} - Accuracy: {accuracy:.3f}")
        print(f"Predictions: {pred_classes}")
        print(f"Raw predictions (first 2 images): {preds[:2]}")
    
    # Compare predictions between models
    print("\nPrediction agreement between models:")
    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            agreement = np.mean(all_preds[model1] == all_preds[model2])
            print(f"{model1} vs {model2}: {agreement:.3f}")
