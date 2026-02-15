import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class SimpleCropDiseaseModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build a simpler CNN model without transfer learning"""
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Simple CNN model built with {self.num_classes} classes")
        self.model.summary()
        
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """Train the model"""
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=8, restore_best_weights=True),
            ModelCheckpoint(
                f'models/simple_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            ReduceLROnPlateau(factor=0.1, patience=4)
        ]
        
        print("Training simple CNN model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
    def evaluate(self, X_test, y_test, classes):
        """Evaluate the model"""
        print("Evaluating model...")
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=classes))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/simple_confusion_matrix.png')
        plt.show()
        
        # Plot training history
        self.plot_training_history()
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('models/simple_training_history.png')
        plt.show()
        
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def train_simple_model():
    """Train simple CNN model for wheat disease classification"""
    print("=== Training Simple CNN Model ===")
    
    # Load processed data
    X_train = np.load('data/wheat/processed/X_train.npy')
    X_val = np.load('data/wheat/processed/X_val.npy')
    X_test = np.load('data/wheat/processed/X_test.npy')
    y_train = np.load('data/wheat/processed/y_train.npy')
    y_val = np.load('data/wheat/processed/y_val.npy')
    y_test = np.load('data/wheat/processed/y_test.npy')
    
    # Load classes
    with open('data/wheat/processed/classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Create and train model
    model = SimpleCropDiseaseModel(num_classes=len(classes))
    model.build_model()
    model.train(X_train, y_train, X_val, y_val)
    model.evaluate(X_test, y_test, classes)
    model.save_model('models/simple_model.h5')
    
    return model, classes

if __name__ == "__main__":
    train_simple_model()
