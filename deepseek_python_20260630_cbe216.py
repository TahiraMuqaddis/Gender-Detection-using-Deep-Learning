# ============================================================
# GENDER DETECTION - COMPLETE TRAINING NOTEBOOK
# ============================================================

# ============================================================
# STEP 1: SETUP AND IMPORTS
# ============================================================

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as ka
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("All libraries imported successfully! ✅")

# ============================================================
# STEP 2: CONFIGURATION
# ============================================================

# Update this path to match your dataset location
DataSet_Path = "/content/drive/MyDrive/dataset/task"

# Paths for train, test, validation
TRAIN_PATH = os.path.join(DataSet_Path, "train")
TEST_PATH = os.path.join(DataSet_Path, "test")
VALID_PATH = os.path.join(DataSet_Path, "valid")

IMG_SIZE = 128  # Image size for training

print("Training path:", TRAIN_PATH)
print("Test path:", TEST_PATH)
print("Validation path:", VALID_PATH)

# ============================================================
# STEP 3: DATA LOADING FUNCTIONS
# ============================================================

def read_image(image_path):
    """Read and preprocess a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Resize to target size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.array(img)

def prepare_data(folder_path):
    """Load all images and labels from a folder."""
    images = []
    labels = []
    skipped = 0
    
    class_folders = os.listdir(folder_path)
    print("Classes Found:", class_folders)
    
    for class_name in class_folders:
        class_path = os.path.join(folder_path, class_name)
        class_lower = class_name.lower()
        
        # Assign labels: women=0, men=1
        if class_lower == "women":
            label = 0
        elif class_lower == "men":
            label = 1
        else:
            print(f"⚠️ Unknown class folder: {class_name}")
            continue
        
        image_files = os.listdir(class_path)
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            # Only process image files
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            img = read_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(label)
            else:
                skipped += 1
        
        print(f"✅ Loaded {len(os.listdir(class_path))} images from {class_name} (Label = {label})")
    
    print(f"⚠️ Skipped {skipped} invalid images")
    return np.array(images), np.array(labels)

# ============================================================
# STEP 4: LOAD DATA
# ============================================================

print("\n" + "="*50)
print("LOADING TRAINING DATA")
print("="*50)
x_train, y_train = prepare_data(TRAIN_PATH)
print(f"Training set: {x_train.shape[0]} images")

print("\n" + "="*50)
print("LOADING TEST DATA")
print("="*50)
x_test, y_test = prepare_data(TEST_PATH)
print(f"Test set: {x_test.shape[0]} images")

print("\n" + "="*50)
print("LOADING VALIDATION DATA")
print("="*50)
x_valid, y_valid = prepare_data(VALID_PATH)
print(f"Validation set: {x_valid.shape[0]} images")

# ============================================================
# STEP 5: DATA EXPLORATION AND VISUALIZATION
# ============================================================

# Display sample images from training data
female_indices = np.where(y_train == 0)[0][:5]
male_indices = np.where(y_train == 1)[0][:5]

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Training Images\n(Row 1: Female | Row 2: Male)", 
             fontsize=14, fontweight='bold')

# Row 1: Female images
for i, idx in enumerate(female_indices):
    axes[0, i].imshow(x_train[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axes[0, i].set_title(f"Female (0)", fontsize=9, color='deeppink')
    axes[0, i].axis('off')

# Row 2: Male images
for i, idx in enumerate(male_indices):
    axes[1, i].imshow(x_train[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axes[1, i].set_title(f"Male (1)", fontsize=9, color='royalblue')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# Check class distribution
print("\nClass Distribution:")
print(f"Training - Female: {np.sum(y_train==0)}, Male: {np.sum(y_train==1)}")
print(f"Test - Female: {np.sum(y_test==0)}, Male: {np.sum(y_test==1)}")
print(f"Validation - Female: {np.sum(y_valid==0)}, Male: {np.sum(y_valid==1)}")

# ============================================================
# STEP 6: DATA PREPROCESSING
# ============================================================

# 6.1: Flatten images (2D → 1D)
print("\n" + "="*50)
print("FLATTENING IMAGES")
print("="*50)

x_train = x_train.reshape(len(x_train), IMG_SIZE * IMG_SIZE)
x_test = x_test.reshape(len(x_test), IMG_SIZE * IMG_SIZE)
x_valid = x_valid.reshape(len(x_valid), IMG_SIZE * IMG_SIZE)

print(f"Training shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Validation shape: {x_valid.shape}")

# 6.2: L2 Normalization
print("\n" + "="*50)
print("L2 NORMALIZATION")
print("="*50)

x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
x_valid = preprocessing.normalize(x_valid)

print("✅ Data normalized successfully!")

# 6.3: Shuffle data
print("\n" + "="*50)
print("SHUFFLING DATA")
print("="*50)

x_train, y_train = shuffle(x_train, y_train, random_state=42)
x_test, y_test = shuffle(x_test, y_test, random_state=42)
x_valid, y_valid = shuffle(x_valid, y_valid, random_state=42)

print("✅ Data shuffled successfully!")

# 6.4: One-hot encode labels
print("\n" + "="*50)
print("ONE-HOT ENCODING LABELS")
print("="*50)

# Save original labels for evaluation
y_train_original = y_train.copy()
y_test_original = y_test.copy()
y_valid_original = y_valid.copy()

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"y_valid shape: {y_valid.shape}")
print("✅ Labels one-hot encoded!")

# ============================================================
# STEP 7: BUILD THE MODEL
# ============================================================

print("\n" + "="*50)
print("BUILDING THE MODEL")
print("="*50)

model = Sequential(name="Gender_Detection", layers=[
    Input(shape=(IMG_SIZE * IMG_SIZE,)),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Add dropout for regularization
    Dense(64, activation='relu'),
    Dropout(0.2),  # Add dropout for regularization
    Dense(2, activation='softmax')
])

print("Model Architecture:")
model.summary()

# ============================================================
# STEP 8: COMPILE THE MODEL
# ============================================================

print("\n" + "="*50)
print("COMPILING THE MODEL")
print("="*50)

model.compile(
    optimizer=ka.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled successfully!")

# ============================================================
# STEP 9: TRAIN THE MODEL
# ============================================================

print("\n" + "="*50)
print("TRAINING THE MODEL")
print("="*50)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_valid, y_valid),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✅ Training completed!")

# ============================================================
# STEP 10: EVALUATE THE MODEL
# ============================================================

print("\n" + "="*50)
print("EVALUATING THE MODEL")
print("="*50)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# STEP 11: PREDICTIONS AND CONFUSION MATRIX
# ============================================================

print("\n" + "="*50)
print("PREDICTIONS AND CONFUSION MATRIX")
print("="*50)

# Make predictions
y_pred_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_original, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_original, y_pred, target_names=['Female', 'Male']))

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')

# Add labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Female', 'Male'])
ax.set_yticklabels(['Female', 'Male'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.show()

# ============================================================
# STEP 12: SAVE THE MODEL
# ============================================================

print("\n" + "="*50)
print("SAVING THE MODEL")
print("="*50)

# Save to Google Drive
save_path = "/content/drive/MyDrive/dataset/gender.keras"
model.save(save_path)
print(f"✅ Model saved to: {save_path}")

# Also save locally (if in Colab)
model.save("gender.keras")
print("✅ Model saved locally as 'gender.keras'")

print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
print("="*50)