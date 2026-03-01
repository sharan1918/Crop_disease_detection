import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def augment_images():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'Full_dataset', 'tomato', 'Healthy')
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: {input_dir} does not exist. Please put your 1,000 tomato healthy images there first.")
        return

    # Initialize data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_count = len(image_files)
    
    if original_count == 0:
        print("❌ No images found in the 'Healthy' folder.")
        return
        
    print(f"📸 Found {original_count} original images. Augmenting to reach 5,000...")

    # We need 5000 images total. 
    # original_count + (original_count * multipliers) = 5000
    # If original_count is 1000, we need 4 augmented copies per image.
    multiplier = int(5000 / original_count) - 1

    progress_bar = tqdm(total=original_count * multiplier, desc="Augmenting")

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((1,) + img.shape)

        i = 0
        name_base = os.path.splitext(filename)[0]
        for batch in datagen.flow(img, batch_size=1, save_to_dir=input_dir, 
                                  save_prefix=f"aug_{name_base}", save_format='jpg'):
            i += 1
            progress_bar.update(1)
            if i >= multiplier:
                break

    progress_bar.close()
    
    final_count = len([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n✅ Augmentation complete! Total images in 'Healthy' folder: {final_count}")

if __name__ == "__main__":
    augment_images()
