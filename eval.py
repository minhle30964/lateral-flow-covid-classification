import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold

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

# Data loading
train_path = os.path.join('/root/covid-classification/MajorProject')
neg_images = os.listdir(train_path + '/Negative')
post_images = os.listdir(train_path + '/Positive')

WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def load_data():
    neg_df = pd.DataFrame({'id': neg_images, 'label': 0})
    post_df = pd.DataFrame({'id': post_images, 'label': 1})
    
    data = []
    for path, df in [(train_path + '/Negative/', neg_df),
                     (train_path + '/Positive/', post_df)]:
        for img_id in df['id']:
            img = Image.open(path + img_id)
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
    
    data = np.array(data)
    labels = pd.concat([neg_df, post_df])['label']
    return data, labels

# Apply difficulty transformations
def apply_difficulty(data, noise_std=0, shadow_alpha=0, rotation_range=0, brightness_range=None):
    # Apply gaussian noise first
    modified_data = data.copy()
    if noise_std > 0:
        modified_data = add_gaussian_noise(modified_data, std=noise_std)
    
    # Apply shadows
    if shadow_alpha > 0:
        modified_data = add_shadows(modified_data, alpha=shadow_alpha)
    
    # Apply rotation and brightness changes if needed
    if rotation_range > 0 or brightness_range is not None:
        # Create a generator with the specified parameters
        datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            brightness_range=brightness_range,
            fill_mode='nearest'
        )
        
        # Process each image individually to avoid batch issues
        augmented = []
        for img in modified_data:
            # Reshape for the generator (add batch dimension)
            img_batch = np.expand_dims(img, 0)
            # Get the augmented image
            aug_iter = datagen.flow(img_batch, batch_size=1, seed=42)
            aug_img = next(aug_iter)[0]
            augmented.append(aug_img)
        
        # Convert back to numpy array
        modified_data = np.array(augmented)
    
    return modified_data

# Load models
models = {
    'Model_A': tf.keras.models.load_model('my_model.h5'),
    'Model_B': tf.keras.models.load_model('my_model-SGD.h5'),
    'Model_C': tf.keras.models.load_model('model-nasnet.h5'),
    'Model_D': tf.keras.models.load_model('densenet-model.h5')
    # 'Model_E': tf.keras.models.load_model('resnet50-model.h5')  # Uncomment if available
}

# Evaluation function with fixed test set
def evaluate_with_fixed_test_set(modified_data, y_test, difficulty_name):
    y_test_cat = keras.utils.to_categorical(y_test)
    
    # Calculate class weights to address imbalance
    class_counts = np.bincount(y_test)
    total_samples = len(y_test)
    class_weights = total_samples / (len(class_counts) * class_counts)
    print(f"Class weights: {class_weights}")
    
    results = {}
    for name, model in models.items():
        pred = model.predict(modified_data, verbose=0)  # Suppress progress bar
        y_pred = np.argmax(pred, axis=1)
        y_true = y_test
        
        # Calculate standard metrics
        acc = np.mean(y_pred == y_true)
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Calculate balanced metrics
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Calculate weighted accuracy using class weights
        weighted_correct = np.sum((y_pred == y_true) * class_weights[y_true])
        weighted_acc = weighted_correct / np.sum(class_weights[y_true])
        
        results[name] = {
            'accuracy': acc, 
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'weighted_accuracy': weighted_acc
        }
        
        print(f"{difficulty_name} - {name}:")
        print(f"  Standard Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")
        print(f"  Balanced Accuracy: {balanced_acc:.3f}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(f"  Weighted Accuracy: {weighted_acc:.3f}")
        print(f"  Class distribution in predictions: {np.bincount(y_pred)}")
    
    return results

# Main experiment
data, labels = load_data()

# Print class distribution
class_distribution = np.bincount(labels)
print(f"Class distribution in dataset: {class_distribution}")
print(f"Class imbalance ratio: 1:{class_distribution[1]/class_distribution[0]:.1f}")

# Create a consistent stratified train/test split to use across all difficulty levels
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels
)

print(f"Train set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# Difficulty levels
difficulties = {
    'Clean': {'noise_std': 0, 'shadow_alpha': 0, 'rotation_range': 0, 'brightness_range': None},
    'Low Noise': {'noise_std': 0.1, 'shadow_alpha': 0.2, 'rotation_range': 10, 'brightness_range': [0.8, 1.2]},
    'Medium Noise': {'noise_std': 0.2, 'shadow_alpha': 0.4, 'rotation_range': 20, 'brightness_range': [0.6, 1.4]},
    'High Noise': {'noise_std': 0.3, 'shadow_alpha': 0.6, 'rotation_range': 30, 'brightness_range': [0.4, 1.6]},
    'Extreme Noise': {'noise_std': 0.5, 'shadow_alpha': 0.8, 'rotation_range': 45, 'brightness_range': [0.2, 1.8]}
}

# Run experiments
all_results = {}

# Diagnostic: Check if models are different
print("\nDiagnostic - Checking model architectures:")
for name, model in models.items():
    print(f"\nModel: {name}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"First layer weights shape: {model.layers[0].get_weights()[0].shape if len(model.layers[0].get_weights()) > 0 else 'No weights'}")

for name, params in difficulties.items():
    print(f"\nTesting {name} difficulty:")
    # Apply difficulty only to the test set
    modified_test_data = apply_difficulty(X_test, **params)
    
    # Diagnostic: Check a sample image before and after noise
    if name != 'Clean':
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(X_test[0])
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(modified_test_data[0])
        plt.title(f'{name} Applied')
        plt.savefig(f'diagnostic_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    all_results[name] = evaluate_with_fixed_test_set(modified_test_data, y_test, name)

# Rank models and analyze
def rank_models(results, metric='balanced_accuracy'):
    for difficulty, model_scores in results.items():
        print(f"\nRanking for {difficulty} ({metric}):")
        ranked = sorted(model_scores.items(), key=lambda x: x[1][metric], reverse=True)
        for i, (model, scores) in enumerate(ranked, 1):
            print(f"{i}. {model}: {metric} = {scores[metric]:.3f}")

rank_models(all_results, 'balanced_accuracy')

# Visualization functions
def plot_f1_scores(all_results, output_dir='graphs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    difficulties = list(all_results.keys())
    model_names = list(models.keys())
    
    # Plot F1 scores per model across difficulties
    plt.figure(figsize=(10, 6))
    for model in model_names:
        f1_scores = [all_results[diff][model]['f1_score'] for diff in difficulties]
        plt.plot(difficulties, f1_scores, marker='o', label=model)
    plt.title('F1 Score Across Difficulty Levels')
    plt.xlabel('Difficulty Level')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_scores_line.png'))
    plt.close()

def plot_rankings_bar(all_results, metric='balanced_accuracy', output_dir='graphs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for difficulty, model_scores in all_results.items():
        models_sorted = sorted(model_scores.items(), key=lambda x: x[1][metric], reverse=True)
        names = [m[0] for m in models_sorted]
        scores = [m[1][metric] for m in models_sorted]
        
        plt.figure(figsize=(8, 5))
        plt.bar(names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(f'Model Rankings - {difficulty} ({metric})')
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1.1)
        for i, v in enumerate(scores):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        plt.savefig(os.path.join(output_dir, f'ranking_{difficulty.lower().replace(" ", "_")}.png'))
        plt.close()

# Generate and save plots
plot_f1_scores(all_results)
plot_rankings_bar(all_results, 'balanced_accuracy')