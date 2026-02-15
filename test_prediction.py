import requests
import os

# Test the API with a sample image
def test_api():
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test with a sample image
    sample_image = "data/wheat/data/Healthy/Corn_Health (1).jpg"
    
    if os.path.exists(sample_image):
        print(f"\nTesting prediction with: {sample_image}")
        
        try:
            with open(sample_image, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/predict", files=files)
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Sample image not found: {sample_image}")

if __name__ == "__main__":
    test_api()
