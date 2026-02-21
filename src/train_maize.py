import os
import json
import logging
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(num_classes):
    """Create a ResNet50 model customized for our number of classes."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Full_dataset', 'Maize')
    models_dir = os.path.join(base_dir, 'Ml_Models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'maize_disease_model.h5')
    json_save_path = os.path.join(models_dir, 'maize_class_indices.json')
    
    batch_size = 32
    img_size = (224, 224)
    
    logger.info(f"Loading data from {data_dir}...")
    
    # Use ImageDataGenerator for augmentation and preprocessing
    # Note: inference.py uses tf.keras.applications.resnet.preprocess_input
    # so we'll use that as the preprocessing function
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save the class indices
    class_indices = train_generator.class_indices
    logger.info(f"Class indices: {class_indices}")
    with open(json_save_path, 'w') as f:
        json.dump(class_indices, f)
        
    num_classes = len(class_indices)
    
    logger.info("Creating model...")
    model, base_model = create_model(num_classes)
    
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    logger.info("Starting Phase 1: Training top layers...")
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    logger.info("Starting Phase 2: Fine-tuning...")
    # Unfreeze some top layers of ResNet50
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    model.compile(
        optimizer=Adam(learning_rate=1e-5), # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    logger.info(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train()
