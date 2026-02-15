import requests
import json
import os

def test_api():
    """Test the crop disease classification API"""
    
    base_url = "http://localhost:5000"
    
    print("=== Testing Crop Disease Classification API ===\n")
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Make sure the API server is running (python src/app.py)")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Test models info endpoint
    print("2. Testing models info endpoint...")
    try:
        response = requests.get(f"{base_url}/models/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test file upload endpoint (with a sample image if available)
    print("3. Testing file upload endpoint...")
    
    # Look for sample images in the data directories
    sample_images = []
    
    # Search for wheat images
    if os.path.exists("data/wheat"):
        for root, dirs, files in os.walk("data/wheat"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_images.append(os.path.join(root, file))
                    break
            if sample_images:
                break
    
    # Search for maize images if no wheat images found
    if not sample_images and os.path.exists("data/maize"):
        for root, dirs, files in os.walk("data/maize"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_images.append(os.path.join(root, file))
                    break
            if sample_images:
                break
    
    if sample_images:
        sample_image = sample_images[0]
        print(f"Using sample image: {sample_image}")
        
        try:
            with open(sample_image, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/predict", files=files)
            
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
    else:
        print("No sample images found. Please download datasets first:")
        print("  python src/download_datasets.py")
        print("  python src/data_preprocessing.py")
        print("  python src/model_training.py")
    
    print("\n" + "="*50 + "\n")
    
    # Test with crop type specification
    if sample_images:
        print("4. Testing with crop type specification...")
        try:
            with open(sample_image, 'rb') as f:
                files = {'file': f}
                data = {'crop_type': 'wheat'}
                response = requests.post(f"{base_url}/predict", files=files, data=data)
            
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
    
    print("\n=== API Testing Complete ===")

if __name__ == "__main__":
    test_api()
