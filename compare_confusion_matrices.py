import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Set fixed random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data loading settings
train_path = os.path.join('/root/covid-classification/MajorProject')
neg_images = os.listdir(train_path + '/Negative')
post_images = os.listdir(train_path + '/Positive')

WIDTH = 224
HEIGHT = 224
CHANNEL = 3

# Load models
models = {
    'Model_A': tf.keras.models.load_model('my_model.h5'),
    'Model_B': tf.keras.models.load_model('my_model-SGD.h5'),
    'Model_C': tf.keras.models.load_model('model-nasnet.h5'),
    'Model_D': tf.keras.models.load_model('densenet-model.h5')
}

def load_data():
    """Load and prepare the dataset"""
    neg_df = pd.DataFrame({'id': neg_images, 'label': 0})
    post_df = pd.DataFrame({'id': post_images, 'label': 1})
    
    data = []
    labels = []
    for path, df in [(train_path + '/Negative/', neg_df),
                     (train_path + '/Positive/', post_df)]:
        for img_id in df['id']:
            img = Image.open(path + img_id)
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
            labels.append(df.loc[df['id'] == img_id, 'label'].values[0])
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def apply_difficulty(data, noise_std=0, shadow_alpha=0, rotation_range=0, brightness_range=None):
    """Apply various transformations to simulate different difficulty levels"""
    # Apply gaussian noise first
    modified_data = data.copy()
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, modified_data.shape)
        modified_data = np.clip(modified_data + noise, 0, 1)
    
    # Apply shadows
    if shadow_alpha > 0:
        # Get the shape of the images
        batch_size, height, width, channels = modified_data.shape
        
        # Create shadow with proper dimensions (batch_size, height, width)
        shadow = np.random.random((batch_size, height, width)) * shadow_alpha
        
        # Expand to match image dimensions (batch_size, height, width, channels)
        shadow = np.expand_dims(shadow, axis=-1)
        shadow = np.repeat(shadow, channels, axis=-1)
        
        modified_data = np.clip(modified_data * (1 - shadow), 0, 1)
    
    # Apply rotation and brightness changes if needed
    if rotation_range > 0 or brightness_range is not None:
        # Create a generator with the specified parameters
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

def generate_model_comparison_matrices():
    """Generate confusion matrices comparing all models on clean data"""
    # Load data
    data, labels = load_data()
    
    # Print class distribution
    class_distribution = np.bincount(labels)
    print(f"Class distribution in dataset: {class_distribution}")
    print(f"Class imbalance ratio: 1:{class_distribution[1]/class_distribution[0]:.1f}")
    
    # Create a consistent stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels
    )
    
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    # Create a directory for saving comparison matrices
    os.makedirs('confusion_matrices', exist_ok=True)
    
    # Create a figure with subplots for all models
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Store metrics for all models
    all_metrics = {}
    
    # Process each model
    for i, (name, model) in enumerate(models.items()):
        # Get predictions
        pred = model.predict(X_test, verbose=0)
        y_pred = np.argmax(pred, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Negative', 'Positive']
        )
        
        # Plot on the corresponding subplot
        disp.plot(
            ax=axes[i],
            cmap='Blues',
            values_format='d',
            colorbar=False,
        )
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative class
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision for positive class
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision for negative class
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        # Store metrics
        all_metrics[name] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1': f1
        }
        
        # Add metrics as text
        metrics_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"Sensitivity: {sensitivity:.3f}\n"
            f"Specificity: {specificity:.3f}\n"
            f"PPV: {ppv:.3f}\n"
            f"NPV: {npv:.3f}\n"
            f"F1: {f1:.3f}"
        )
        
        axes[i].set_title(f"{name}")
        axes[i].text(1.05, 0.5, metrics_text, transform=axes[i].transAxes, 
                    verticalalignment='center', fontsize=10)
    
    plt.suptitle("Confusion Matrices for Models A-D on Clean Data", fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrices/model_comparison_clean.png', dpi=300)
    plt.close()
    
    # Create a metrics comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the table
    model_names = list(all_metrics.keys())
    metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 Score']
    
    # Create the table data
    table_data = []
    for metric, metric_name in zip(metrics, metric_names):
        row = [metric_name]
        for model in model_names:
            row.append(f"{all_metrics[model][metric]:.3f}")
        table_data.append(row)
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric'] + model_names,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(model_names) + 1):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white')
    
    plt.title('Performance Metrics Comparison', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('confusion_matrices/metrics_comparison_table.png', dpi=300)
    plt.close()
    
    print("Model comparison matrices and metrics table saved.")

def generate_difficulty_comparison_matrices():
    """Generate confusion matrices comparing difficulty levels for each model"""
    # Load data
    data, labels = load_data()
    
    # Create a consistent stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels
    )
    
    # Define difficulty levels
    difficulties = {
        'Clean': {'noise_std': 0, 'shadow_alpha': 0, 'rotation_range': 0, 'brightness_range': None},
        'Low Noise': {'noise_std': 0.1, 'shadow_alpha': 0.2, 'rotation_range': 10, 'brightness_range': [0.8, 1.2]},
        'Medium Noise': {'noise_std': 0.2, 'shadow_alpha': 0.4, 'rotation_range': 20, 'brightness_range': [0.6, 1.4]},
        'High Noise': {'noise_std': 0.3, 'shadow_alpha': 0.6, 'rotation_range': 30, 'brightness_range': [0.4, 1.6]},
        'Extreme Noise': {'noise_std': 0.5, 'shadow_alpha': 0.8, 'rotation_range': 45, 'brightness_range': [0.2, 1.8]}
    }
    
    # Create a directory for saving comparison matrices
    os.makedirs('confusion_matrices', exist_ok=True)
    
    # For each model, create a figure comparing performance across difficulty levels
    for model_name, model in models.items():
        # Create a figure with subplots for all difficulty levels
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Store metrics for all difficulty levels
        difficulty_metrics = {}
        
        # Process each difficulty level
        for i, (diff_name, params) in enumerate(difficulties.items()):
            if i >= len(axes):  # Skip if we run out of subplots
                continue
                
            # Apply difficulty transformations to test data
            modified_test_data = apply_difficulty(X_test, **params)
            
            # Get predictions
            pred = model.predict(modified_test_data, verbose=0)
            y_pred = np.argmax(pred, axis=1)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Display confusion matrix
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Negative', 'Positive']
            )
            
            # Plot on the corresponding subplot
            disp.plot(
                ax=axes[i],
                cmap='Blues',
                values_format='d',
                colorbar=False,
            )
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive class
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative class
            
            # Store metrics
            difficulty_metrics[diff_name] = {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
            }
            
            # Add metrics as text
            metrics_text = (
                f"Accuracy: {accuracy:.3f}\n"
                f"Sensitivity: {sensitivity:.3f}\n"
                f"Specificity: {specificity:.3f}"
            )
            
            axes[i].set_title(f"{diff_name}")
            axes[i].text(1.05, 0.5, metrics_text, transform=axes[i].transAxes, 
                        verticalalignment='center', fontsize=10)
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f"Confusion Matrices for {model_name} Across Difficulty Levels", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'confusion_matrices/{model_name}_difficulty_comparison.png', dpi=300)
        plt.close()
        
        print(f"Difficulty comparison matrices for {model_name} saved.")

if __name__ == "__main__":
    # Generate model comparison on clean data
    generate_model_comparison_matrices()
    
    # Generate difficulty comparison for each model
    generate_difficulty_comparison_matrices()
