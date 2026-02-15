import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CropDiseaseModel:
    def __init__(self, model_type="resnet50", input_shape=(224, 224, 3), num_classes=None):
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN model using transfer learning"""
        
        if self.model_type == "resnet50":
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_type == "vgg16":
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_type == "mobilenetv2":
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the complete model
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Preprocessing for the base model
        if self.model_type == "resnet50":
            x = tf.keras.applications.resnet.preprocess_input(x)
        elif self.model_type == "vgg16":
            x = tf.keras.applications.vgg16.preprocess_input(x)
        elif self.model_type == "mobilenetv2":
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built using {self.model_type}")
        self.model.summary()
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f'models/{self.model_type}_best.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            ReduceLROnPlateau(factor=0.1, patience=5)
        ]
        
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning: unfreeze some layers
        print("Fine-tuning model...")
        
        # Unfreeze the top layers of the base model
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        # Re-compile with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        self.history_fine = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs//2,
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
        plt.savefig('models/confusion_matrix.png')
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
        if hasattr(self, 'history_fine'):
            ax1.plot(self.history_fine.history['accuracy'], label='Fine-tuning Accuracy')
            ax1.plot(self.history_fine.history['val_accuracy'], label='Fine-tuning Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        if hasattr(self, 'history_fine'):
            ax2.plot(self.history_fine.history['loss'], label='Fine-tuning Loss')
            ax2.plot(self.history_fine.history['val_loss'], label='Fine-tuning Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
        
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def train_wheat_model():
    """Train wheat disease classification model"""
    print("=== Training Wheat Disease Model ===")
    
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
    model = CropDiseaseModel(model_type="resnet50", num_classes=len(classes))
    model.build_model()
    model.train(X_train, y_train, X_val, y_val)
    model.evaluate(X_test, y_test, classes)
    model.save_model('models/wheat_disease_model.h5')
    
    return model, classes

def train_maize_model():
    """Train maize disease classification model"""
    print("=== Training Maize Disease Model ===")
    
    # Load processed data
    X_train = np.load('data/maize/processed/X_train.npy')
    X_val = np.load('data/maize/processed/X_val.npy')
    X_test = np.load('data/maize/processed/X_test.npy')
    y_train = np.load('data/maize/processed/y_train.npy')
    y_val = np.load('data/maize/processed/y_val.npy')
    y_test = np.load('data/maize/processed/y_test.npy')
    
    # Load classes
    with open('data/maize/processed/classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Create and train model
    model = CropDiseaseModel(model_type="resnet50", num_classes=len(classes))
    model.build_model()
    model.train(X_train, y_train, X_val, y_val)
    model.evaluate(X_test, y_test, classes)
    model.save_model('models/maize_disease_model.h5')
    
    return model, classes

def main():
    """Main training function"""
    
    # Train wheat model
    wheat_model, wheat_classes = train_wheat_model()
    
    # Train maize model
    maize_model, maize_classes = train_maize_model()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
