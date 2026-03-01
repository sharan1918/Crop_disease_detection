import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import sys

def augment_folder(target_folder, target_total=5000):
    if not os.path.exists(target_folder):
        print(f"❌ Error: {target_folder} does not exist.")
        return

    # Initialize data generator
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_count = len(image_files)
    
    if original_count == 0:
        print(f"❌ No images found in {target_folder}.")
        return

    if original_count >= target_total:
        print(f"✅ Folder already has {original_count} images (target is {target_total}). Skipping.")
        return
        
    print(f"📸 Found {original_count} images in {os.path.basename(target_folder)}. Augmenting to reach {target_total}...")

    # Calculate how many extra images we need
    needed = target_total - original_count
    
    # How many times to loop through each image
    multiplier = needed // original_count
    remainder = needed % original_count
    
    progress_bar = tqdm(total=needed, desc=f"Augmenting {os.path.basename(target_folder)}")

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(target_folder, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((1,) + img.shape)

        # Determine how many for this specific image
        count_for_this = multiplier + (1 if idx < remainder else 0)
        if count_for_this <= 0: continue

        i = 0
        name_base = os.path.splitext(filename)[0]
        # Use a high save batch size or just loop
        for batch in datagen.flow(img, batch_size=1, save_to_dir=target_folder, 
                                  save_prefix=f"aug_{name_base}", save_format='jpg'):
            i += 1
            progress_bar.update(1)
            if i >= count_for_this:
                break

    progress_bar.close()
    
    final_count = len([f for f in os.listdir(target_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"✅ Finished! Total in {os.path.basename(target_folder)}: {final_count}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # List of folders to augment based on user choice
    folders = [
        os.path.join(base_dir, 'Full_dataset', 'tomato', 'Unhealthy', 'Leaf Blight'),
        os.path.join(base_dir, 'Full_dataset', 'tomato', 'Unhealthy', 'Bacterial Spot'),
    ]

    for folder in folders:
        if os.path.exists(folder):
            augment_folder(folder, 5000)
        else:
            print(f"⚠️ Folder not found: {folder}. Please move your images there first.")

if __name__ == "__main__":
    main()
