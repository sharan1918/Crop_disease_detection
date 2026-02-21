import os
import requests
import concurrent.futures
import time
from tqdm import tqdm

def download_image(url, save_path, attempt=1):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        if attempt < 3:
            time.sleep(1)
            return download_image(url, save_path, attempt + 1)
        return False

def main():
    # Target directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, 'Full_dataset', 'Maize', 'Others')
    os.makedirs(target_dir, exist_ok=True)
    
    # We want around 800 images to roughly balance with Blight (870) and Healthy (we'll augment or use class weights)
    # Actually, let's download 800 random images from Lorem Picsum
    num_images = 800
    print(f"Downloading {num_images} random images to {target_dir}...")
    
    # Generate URLs (using a random seed in the URL to ensure different images)
    tasks = []
    for i in range(num_images):
        url = f"https://picsum.photos/seed/{i}/224/224"
        save_path = os.path.join(target_dir, f"random_{i}.jpg")
        tasks.append((url, save_path))
    
    # Download concurrently to speed things up
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_image, url, path): (url, path) for url, path in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_images, desc="Downloading images"):
            if future.result():
                success_count += 1
                
    print(f"\nSuccessfully downloaded {success_count}/{num_images} images.")

if __name__ == "__main__":
    main()
