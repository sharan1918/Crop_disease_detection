#!/usr/bin/env python3
"""
Download and organize Wheat Plant Diseases Dataset
Downloads from Kaggle and organizes into 7 classes in the wheat folder
"""

import os
import shutil
import zipfile
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kaggle_dataset():
    """Download wheat plant diseases dataset from Kaggle"""
    
    # Target directories
    base_dir = Path(r"C:\Users\Vijay\OneDrive\Desktop\Crop_disease_detection")
    wheat_dir = base_dir / "Full_dataset" / "wheat"
    download_dir = base_dir / "downloads"
    
    # Create directories
    wheat_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(exist_ok=True)
    
    # Create class directories
    classes = ["Healthy", "Black_Rust", "Brown_Rust", "Yellow_Rust", "Aphid", "Mite", "Stem_Fly"]
    
    for class_name in classes:
        (wheat_dir / class_name).mkdir(exist_ok=True)
    
    logger.info("Created directory structure for wheat dataset")
    
    # Note: Kaggle datasets require API authentication
    # This script provides the structure - you'll need to:
    # 1. Install kaggle: pip install kaggle
    # 2. Get API token from kaggle.com/account
    # 3. Place token in C:\Users\Vijay\.kaggle\kaggle.json
    
    logger.info("""
    MANUAL DOWNLOAD INSTRUCTIONS:
    
    1. Go to: https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases
    2. Click "Download" button
    3. Save to: C:\\Users\\Vijay\\OneDrive\\Desktop\\Crop_disease_detection\\downloads\\
    4. Extract the zip file
    5. Run the organize_wheat_dataset() function below
    
    Or use Kaggle API:
    - Install: pip install kaggle
    - Get API token from kaggle.com/account
    - Run: kaggle datasets download -d kushagra3204/wheat-plant-diseases -p downloads/
    """)

def organize_wheat_dataset():
    """Organize downloaded wheat dataset into the 7 classes"""
    
    base_dir = Path(r"C:\Users\Vijay\OneDrive\Desktop\Crop_disease_detection")
    wheat_dir = base_dir / "Full_dataset" / "wheat"
    download_dir = base_dir / "downloads"
    
    # Expected downloaded dataset structure
    downloaded_dataset = download_dir / "wheat-plant-diseases"
    
    if not downloaded_dataset.exists():
        logger.error(f"Downloaded dataset not found at {downloaded_dataset}")
        logger.info("Please download and extract the dataset first!")
        return False
    
    # Class mapping from dataset to our structure
    class_mapping = {
        "Healthy": "Healthy",
        "Black Rust": "Black_Rust", 
        "Brown Rust": "Brown_Rust",
        "Yellow Rust": "Yellow_Rust",
        "Aphid": "Aphid",
        "Mite": "Mite", 
        "Stem Fly": "Stem_Fly"
    }
    
    # Alternative mapping (if dataset uses different folder names)
    alt_mapping = {
        "Healthy": "Healthy",
        "Black_Rust": "Black_Rust",
        "Brown_Rust": "Brown_Rust", 
        "Yellow_Rust": "Yellow_Rust",
        "Aphid": "Aphid",
        "Mite": "Mite",
        "Stem_Fly": "Stem_Fly"
    }
    
    # Copy images to organized structure
    total_images = 0
    
    # Try both mappings
    mappings_to_try = [class_mapping, alt_mapping]
    
    for mapping in mappings_to_try:
        for source_class, target_class in mapping.items():
            source_path = downloaded_dataset / source_class
            target_path = wheat_dir / target_class
            
            if source_path.exists():
                # Copy all images from source to target
                image_files = list(source_path.glob("*"))
                for img_file in image_files:
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                        try:
                            shutil.copy2(img_file, target_path / img_file.name)
                            total_images += 1
                        except Exception as e:
                            logger.warning(f"Could not copy {img_file}: {e}")
                
                logger.info(f"Copied {len(image_files)} images from {source_class} to {target_class}")
    
    logger.info(f"Total images organized: {total_images}")
    
    # Show final structure
    logger.info("\nFinal wheat dataset structure:")
    for class_name in ["Healthy", "Black_Rust", "Brown_Rust", "Yellow_Rust", "Aphid", "Mite", "Stem_Fly"]:
        class_dir = wheat_dir / class_name
        image_count = len(list(class_dir.glob("*")))
        logger.info(f"  {class_name}: {image_count} images")
    
    return True

def verify_dataset():
    """Verify the wheat dataset is properly organized"""
    
    wheat_dir = Path(r"C:\Users\Vijay\OneDrive\Desktop\Crop_disease_detection\Full_dataset\wheat")
    
    classes = ["Healthy", "Black_Rust", "Brown_Rust", "Yellow_Rust", "Aphid", "Mite", "Stem_Fly"]
    
    logger.info("Wheat Dataset Verification:")
    total_images = 0
    
    for class_name in classes:
        class_dir = wheat_dir / class_name
        if class_dir.exists():
            image_count = len([f for f in class_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']])
            total_images += image_count
            logger.info(f"  ✅ {class_name}: {image_count} images")
        else:
            logger.info(f"  ❌ {class_name}: Directory not found")
    
    logger.info(f"\nTotal images: {total_images}")
    
    if total_images > 0:
        logger.info("✅ Wheat dataset is ready for training!")
    else:
        logger.info("❌ No images found. Please download and organize the dataset first.")

if __name__ == "__main__":
    print("🌾 Wheat Dataset Download and Organization Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Show download instructions")
        print("2. Organize downloaded dataset") 
        print("3. Verify dataset")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            download_kaggle_dataset()
        elif choice == "2":
            organize_wheat_dataset()
        elif choice == "3":
            verify_dataset()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")
