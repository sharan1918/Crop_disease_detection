import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import pickle

def train_quick_model():
    """Train a quick model with minimal data"""
    print("=== Quick Training with 100 Images ===")
    
    # Load maize data (since corn is now in maize folder)
    try:
        X_train = np.load('data/maize/processed/X_train.npy')
        X_val = np.load('data/maize/processed/X_val.npy')
        X_test = np.load('data/maize/processed/X_test.npy')
        y_train = np.load('data/maize/processed/y_train.npy')
        y_val = np.load('data/maize/processed/y_val.npy')
        y_test = np.load('data/maize/processed/y_test.npy')
        
        with open('data/maize/processed/classes.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            
        print(f"Loaded {len(X_train)} training images")
        print(f"Classes: {classes}")
        
    except FileNotFoundError:
        print("Maize data not found. Running preprocessing first...")
        os.system("uv run python src/data_preprocessing.py")
        
        # Try loading again
        X_train = np.load('data/maize/processed/X_train.npy')
        X_val = np.load('data/maize/processed/X_val.npy')
        X_test = np.load('data/maize/processed/X_test.npy')
        y_train = np.load('data/maize/processed/y_train.npy')
        y_val = np.load('data/maize/processed/y_val.npy')
        y_test = np.load('data/maize/processed/y_test.npy')
        
        with open('data/maize/processed/classes.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    
    # Build a very simple model
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(classes), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model built. Training...")
    model.summary()
    
    # Train for just 5 epochs for speed
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/quick_maize_model.h5')
    print("Quick model saved to models/quick_maize_model.h5")
    
    return model, classes

if __name__ == "__main__":
    train_quick_model()
