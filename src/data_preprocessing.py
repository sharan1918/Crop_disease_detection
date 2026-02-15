import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil

class DataPreprocessor:
    def __init__(self, data_dir, image_size=(224, 224)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.classes = []
        self.label_encoder = LabelEncoder()
        
    def explore_dataset(self):
        """Explore the dataset structure and class distribution"""
        print(f"Exploring dataset in: {self.data_dir}")
        
        class_counts = {}
        for root, dirs, files in os.walk(self.data_dir):
            if files:
                class_name = os.path.basename(root)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    class_counts[class_name] = len(image_files)
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
        self.classes = list(class_counts.keys())
        return class_counts
    
    def load_and_preprocess_images(self, max_samples_per_class=None):
        """Load and preprocess images from dataset"""
        images = []
        labels = []
        
        print("Loading and preprocessing images...")
        
        for class_name in self.classes:
            class_path = None
            for root, dirs, files in os.walk(self.data_dir):
                if os.path.basename(root) == class_name and files:
                    class_path = root
                    break
            
            if class_path is None:
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            print(f"Processing {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.image_size)
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets"""
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # First split: separate test set
        X_train, X_test, y_train, y_test = train_test_split(
            images, encoded_labels, test_size=test_size, 
            random_state=42, stratify=encoded_labels
        )
        
        # Second split: separate validation set from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, 
            random_state=42, stratify=y_train
        )
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, save_dir):
        """Save processed data to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        
        # Save label encoder and classes
        import pickle
        with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(os.path.join(save_dir, 'classes.txt'), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        
        print(f"Processed data saved to {save_dir}")

def main():
    """Main preprocessing pipeline"""
    
    # Process wheat dataset
    print("=== Processing Wheat Dataset ===")
    wheat_preprocessor = DataPreprocessor("data/wheat")
    wheat_class_counts = wheat_preprocessor.explore_dataset()
    
    if wheat_class_counts:
        wheat_images, wheat_labels = wheat_preprocessor.load_and_preprocess_images(max_samples_per_class=25)  # 25 per class = 100 total
        X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w = wheat_preprocessor.split_data(
            wheat_images, wheat_labels
        )
        wheat_preprocessor.save_processed_data(
            X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w,
            "data/wheat/processed"
        )
    
    # Process maize dataset
    print("\n=== Processing Maize Dataset ===")
    maize_preprocessor = DataPreprocessor("data/maize")
    maize_class_counts = maize_preprocessor.explore_dataset()
    
    if maize_class_counts:
        maize_images, maize_labels = maize_preprocessor.load_and_preprocess_images(max_samples_per_class=25)  # 25 per class = 100 total
        X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m = maize_preprocessor.split_data(
            maize_images, maize_labels
        )
        maize_preprocessor.save_processed_data(
            X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m,
            "data/maize/processed"
        )

if __name__ == "__main__":
    main()
