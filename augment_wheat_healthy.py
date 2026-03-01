#!/usr/bin/env python3
"""
Augment wheat healthy images to reach 5,000 images
Uses various augmentation techniques
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WheatHealthyAugmenter:
    def __init__(self, source_dir, target_dir, target_count=5000):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_count = target_count
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def load_images(self):
        """Load all original healthy wheat images"""
        images = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        for img_file in self.source_dir.glob("*"):
            if img_file.suffix.lower() in valid_extensions:
                img = cv2.imread(str(img_file))
                if img is not None:
                    images.append(img)
                    logger.info(f"Loaded: {img_file.name}")
        
        logger.info(f"Total original images: {len(images)}")
        return images
    
    def augment_image(self, img, augmentation_type):
        """Apply specific augmentation to image"""
        h, w = img.shape[:2]
        
        if augmentation_type == "rotate":
            angle = random.uniform(-30, 30)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, matrix, (w, h))
        
        elif augmentation_type == "flip_horizontal":
            return cv2.flip(img, 1)
        
        elif augmentation_type == "flip_vertical":
            return cv2.flip(img, 0)
        
        elif augmentation_type == "brightness":
            brightness = random.uniform(0.7, 1.3)
            return cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        elif augmentation_type == "contrast":
            contrast = random.uniform(0.7, 1.3)
            return cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        elif augmentation_type == "blur":
            kernel_size = random.choice([3, 5])
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        elif augmentation_type == "noise":
            noise = np.random.normal(0, 25, img.shape)
            noisy_img = img + noise
            return np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        elif augmentation_type == "crop":
            # Random crop and resize
            crop_h = int(h * 0.8)
            crop_w = int(w * 0.8)
            start_h = random.randint(0, h - crop_h)
            start_w = random.randint(0, w - crop_w)
            cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
            return cv2.resize(cropped, (w, h))
        
        return img
    
    def generate_augmented_dataset(self):
        """Generate augmented images to reach target count"""
        original_images = self.load_images()
        
        if not original_images:
            logger.error("No original images found!")
            return
        
        # Copy original images first
        for i, img in enumerate(original_images):
            output_path = self.target_dir / f"original_{i:04d}.jpg"
            cv2.imwrite(str(output_path), img)
        
        current_count = len(original_images)
        logger.info(f"Copied {current_count} original images")
        
        augmentation_types = [
            "rotate", "flip_horizontal", "flip_vertical", 
            "brightness", "contrast", "blur", "noise", "crop"
        ]
        
        # Generate augmented images
        while current_count < self.target_count:
            # Select random original image
            original_img = random.choice(original_images)
            
            # Apply random augmentation
            aug_type = random.choice(augmentation_types)
            augmented_img = self.augment_image(original_img, aug_type)
            
            # Save augmented image
            output_path = self.target_dir / f"aug_{current_count:04d}_{aug_type}.jpg"
            cv2.imwrite(str(output_path), augmented_img)
            
            current_count += 1
            
            if current_count % 500 == 0:
                logger.info(f"Generated {current_count}/{self.target_count} images")
        
        logger.info(f"✅ Successfully generated {current_count} healthy wheat images!")
        
        # Show final count
        final_count = len(list(self.target_dir.glob("*.jpg")))
        logger.info(f"Final image count: {final_count}")

def main():
    base_dir = Path(r"C:\Users\Vijay\OneDrive\Desktop\Crop_disease_detection\Full_dataset\wheat")
    
    source_healthy = base_dir / "Healthy"
    target_healthy = base_dir / "Healthy_Augmented"
    
    if not source_healthy.exists():
        logger.error(f"Source directory not found: {source_healthy}")
        return
    
    logger.info("🌾 Wheat Healthy Image Augmentation")
    logger.info(f"Source: {source_healthy}")
    logger.info(f"Target: {target_healthy}")
    logger.info(f"Target count: 5,000 images")
    
    augmenter = WheatHealthyAugmenter(source_healthy, target_healthy, 5000)
    augmenter.generate_augmented_dataset()

if __name__ == "__main__":
    main()
