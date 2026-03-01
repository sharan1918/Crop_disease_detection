import os
import requests
import concurrent.futures
import time
from tqdm import tqdm

def download_image(url, save_path, attempt=1):
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        if attempt < 3:
            time.sleep(2)
            return download_image(url, save_path, attempt + 1)
        return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, 'Full_dataset', 'tomato', 'Others')
    
    # Create required directories
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Full_dataset', 'tomato', 'Healthy'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Full_dataset', 'tomato', 'Unhealthy', 'Disease_1'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Full_dataset', 'tomato', 'Unhealthy', 'Disease_2'), exist_ok=True)

    num_images = 5000
    print(f"🚀 Starting download of {num_images} random (non-crop) images for Tomato 'Others' class.")
    print(f"📁 Target: {target_dir}\n")

    tasks = []
    for i in range(num_images):
        url = f"https://picsum.photos/seed/tomato_others_{i}/224/224"
        save_path = os.path.join(target_dir, f"others_{i}.jpg")
        tasks.append((url, save_path))

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_image, url, path): (url, path) for url, path in tasks}

        for future in tqdm(concurrent.futures.as_completed(futures), total=num_images, desc="Downloading"):
            if future.result():
                success_count += 1

    print(f"\n✅ Successfully downloaded {success_count}/{num_images} images.")
    print(f"📍 Location: {target_dir}")

if __name__ == "__main__":
    main()
