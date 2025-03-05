import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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

# Load models
models = {
    'Model_A': tf.keras.models.load_model('my_model.h5'),
    'Model_B': tf.keras.models.load_model('my_model-SGD.h5'),
    'Model_C': tf.keras.models.load_model('model-nasnet.h5'),
    'Model_D': tf.keras.models.load_model('densenet-model.h5')
}

# Balanced evaluation approaches
def evaluate_with_balanced_sampling(X_test, y_test, difficulty_name, sampling_strategy='undersample'):
    """
    Evaluate models using balanced sampling techniques
    
    Parameters:
    -----------
    X_test : numpy array
        Test images
    y_test : numpy array
        Test labels
    difficulty_name : str
        Name of difficulty level
    sampling_strategy : str
        'undersample', 'oversample', or 'none'
    """
    # Apply balancing if requested
    if sampling_strategy == 'undersample':
        print(f"Applying undersampling - Original distribution: {np.bincount(y_test)}")
        rus = RandomUnderSampler(random_state=42)
        # Reshape images to 2D for sampling
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_test_flat, y_test = rus.fit_resample(X_test_flat, y_test)
        # Reshape back to original shape
        X_test = X_test_flat.reshape(-1, HEIGHT, WIDTH, CHANNEL)
        print(f"After undersampling: {np.bincount(y_test)}")
    
    elif sampling_strategy == 'oversample':
        print(f"Applying SMOTE oversampling - Original distribution: {np.bincount(y_test)}")
        # Reshape images to 2D for SMOTE
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        # Use k_neighbors=min(5, n_samples-1) to avoid the error
        n_minority = np.bincount(y_test)[0]  # Count of minority class
        k_neighbors = min(n_minority-1, 5)  # Use at most n_minority-1 neighbors
        if k_neighbors < 1:
            print("Not enough minority samples for SMOTE. Using class weights only.")
            sampling_strategy = 'none'
        else:
            try:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_test_flat, y_test = smote.fit_resample(X_test_flat, y_test)
                # Reshape back to original shape
                X_test = X_test_flat.reshape(-1, HEIGHT, WIDTH, CHANNEL)
                print(f"After oversampling: {np.bincount(y_test)}")
            except Exception as e:
                print(f"SMOTE failed: {e}. Using class weights only.")
                sampling_strategy = 'none'
    
    # Apply difficulty transformations
    modified_test_data = apply_difficulty(X_test, **difficulties[difficulty_name])
    
    # Calculate class weights
    class_counts = np.bincount(y_test)
    total_samples = len(y_test)
    class_weights = total_samples / (len(class_counts) * class_counts)
    print(f"Class weights: {class_weights}")
    
    results = {}
    for name, model in models.items():
        pred = model.predict(modified_test_data, verbose=0)
        y_pred = np.argmax(pred, axis=1)
        
        # Calculate metrics
        acc = np.mean(y_pred == y_test)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Calculate weighted accuracy
        weighted_correct = np.sum((y_pred == y_test) * class_weights[y_test])
        weighted_acc = weighted_correct / np.sum(class_weights[y_test])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'weighted_accuracy': weighted_acc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': pred
        }
        
        # Print results
        print(f"\n{difficulty_name} - {name} ({sampling_strategy}):")
        print(f"  Standard Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")
        print(f"  Balanced Accuracy: {balanced_acc:.3f}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
        print(f"  Weighted Accuracy: {weighted_acc:.3f}")
        print(f"  Confusion Matrix:\n{cm}")
        print(f"  Class distribution in predictions: {np.bincount(y_pred)}")
    
    return results

# Main experiment
data, labels = load_data()

# Print class distribution
class_distribution = np.bincount(labels)
print(f"Class distribution in dataset: {class_distribution}")
print(f"Class imbalance ratio: 1:{class_distribution[1]/class_distribution[0]:.1f}")

# Create a consistent stratified train/test split
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

# Run experiments with different balancing strategies
sampling_strategies = ['none', 'undersample', 'weighted']
all_results = {}

for strategy in sampling_strategies:
    print(f"\n\n===== Testing with {strategy} strategy =====")
    all_results[strategy] = {}
    
    for name, params in difficulties.items():
        print(f"\nTesting {name} difficulty with {strategy}:")
        
        if strategy == 'weighted':
            # Just use class weights without resampling
            # Apply difficulty transformations directly
            modified_test_data = apply_difficulty(X_test, **params)
            
            # Calculate class weights
            class_counts = np.bincount(y_test)
            total_samples = len(y_test)
            class_weights = total_samples / (len(class_counts) * class_counts)
            print(f"Using class weights: {class_weights}")
            
            results = {}
            for model_name, model in models.items():
                pred = model.predict(modified_test_data, verbose=0)
                y_pred = np.argmax(pred, axis=1)
                
                # Apply threshold adjustment to predictions
                # This can help with class imbalance by adjusting decision boundary
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    # For binary classification, favor the minority class by lowering threshold
                    # Default is 0.5, we'll use a lower threshold to favor minority class
                    threshold = 0.3  # Adjust this based on validation results
                    y_pred_adjusted = (pred[:, 0] >= threshold).astype(int)
                    
                    # Calculate metrics with both standard and adjusted predictions
                    acc = np.mean(y_pred == y_test)
                    acc_adjusted = np.mean(y_pred_adjusted == y_test)
                    
                    balanced_acc = balanced_accuracy_score(y_test, y_pred)
                    balanced_acc_adjusted = balanced_accuracy_score(y_test, y_pred_adjusted)
                    
                    f1 = f1_score(y_test, y_pred, average='binary')
                    f1_adjusted = f1_score(y_test, y_pred_adjusted, average='binary')
                    
                    # Store both results
                    results[model_name] = {
                        'accuracy': acc,
                        'accuracy_adjusted': acc_adjusted,
                        'balanced_accuracy': balanced_acc,
                        'balanced_accuracy_adjusted': balanced_acc_adjusted,
                        'f1_score': f1,
                        'f1_score_adjusted': f1_adjusted,
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'confusion_matrix_adjusted': confusion_matrix(y_test, y_pred_adjusted),
                        'predictions': y_pred,
                        'predictions_adjusted': y_pred_adjusted,
                        'probabilities': pred
                    }
                    
                    print(f"\n{name} - {model_name} (weighted with threshold adjustment):")
                    print(f"  Standard Metrics:")
                    print(f"    Accuracy: {acc:.3f}, Balanced Accuracy: {balanced_acc:.3f}, F1: {f1:.3f}")
                    print(f"    Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                    print(f"  Adjusted Threshold Metrics (threshold={threshold}):")
                    print(f"    Accuracy: {acc_adjusted:.3f}, Balanced Accuracy: {balanced_acc_adjusted:.3f}, F1: {f1_adjusted:.3f}")
                    print(f"    Confusion Matrix:\n{confusion_matrix(y_test, y_pred_adjusted)}")
                    print(f"    Class distribution in adjusted predictions: {np.bincount(y_pred_adjusted)}")
                else:
                    # For non-binary outputs, just use standard metrics
                    results[model_name] = evaluate_with_balanced_sampling(
                        X_test, y_test, name, sampling_strategy='none'
                    )[model_name]
            
            all_results[strategy][name] = results
        else:
            # Use standard sampling approach
            all_results[strategy][name] = evaluate_with_balanced_sampling(
                X_test, y_test, name, sampling_strategy=strategy
            )

# Rank models for each strategy
def rank_models(results, metric='balanced_accuracy'):
    for strategy, strategy_results in results.items():
        print(f"\n\n===== Rankings for {strategy} strategy =====")
        
        for difficulty, model_scores in strategy_results.items():
            print(f"\nRanking for {difficulty} ({metric}):")
            ranked = sorted(model_scores.items(), key=lambda x: x[1][metric], reverse=True)
            for i, (model, scores) in enumerate(ranked, 1):
                print(f"{i}. {model}: {metric} = {scores[metric]:.3f}")

# Generate visualizations
def plot_metrics_comparison(all_results, metric='balanced_accuracy', output_dir='balanced_graphs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot comparison across strategies
    plt.figure(figsize=(15, 10))
    
    for i, strategy in enumerate(all_results.keys()):
        difficulties = list(all_results[strategy].keys())
        
        for model_name in all_results[strategy][difficulties[0]].keys():
            scores = [all_results[strategy][diff][model_name][metric] for diff in difficulties]
            plt.plot(difficulties, scores, marker='o', label=f"{model_name} ({strategy})")
    
    plt.title(f'Model {metric} Comparison Across Strategies')
    plt.xlabel('Difficulty Level')
    plt.ylabel(metric.capitalize().replace('_', ' '))
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
    plt.close()
    
    # Plot confusion matrices for the best strategy
    best_strategy = max(all_results.keys(), key=lambda s: np.mean([
        all_results[s][diff][model][metric] 
        for diff in all_results[s] 
        for model in all_results[s][diff]
    ]))
    
    for difficulty in all_results[best_strategy]:
        plt.figure(figsize=(15, 10))
        
        for i, (model_name, results) in enumerate(all_results[best_strategy][difficulty].items()):
            cm = results['confusion_matrix']
            plt.subplot(2, 2, i+1)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{model_name} - {difficulty}')
            plt.colorbar()
            
            classes = ['Negative', 'Positive']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrices_{difficulty.lower().replace(" ", "_")}.png'))
        plt.close()

# Run analysis
rank_models(all_results)
plot_metrics_comparison(all_results, 'balanced_accuracy')
plot_metrics_comparison(all_results, 'f1_score')
plot_metrics_comparison(all_results, 'precision')
plot_metrics_comparison(all_results, 'recall')

print("\nEvaluation complete. Results saved to balanced_graphs directory.")
