import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

def download_kaggle_dataset(dataset_name, download_path, extract_to):
    """
    Download and extract a dataset from Kaggle
    """
    print(f"Downloading {dataset_name}...")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)
    
    # Find the downloaded zip file
    zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {download_path}")
    
    zip_file = os.path.join(download_path, zip_files[0])
    print(f"Extracting {zip_file}...")
    
    # Extract the dataset
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Remove the zip file
    os.remove(zip_file)
    print(f"Successfully extracted to {extract_to}")

def main():
    """Main function to download all datasets"""
    
    # Create download directory
    download_dir = "downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    # Datasets to download
    datasets = [
        {
            "name": "kushagra3204/wheat-plant-diseases",
            "download_path": os.path.join(download_dir, "wheat_temp"),
            "extract_to": "data/wheat"
        },
        {
            "name": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
            "download_path": os.path.join(download_dir, "maize_temp"),
            "extract_to": "data/maize"
        }
    ]
    
    for dataset in datasets:
        try:
            os.makedirs(dataset["download_path"], exist_ok=True)
            os.makedirs(dataset["extract_to"], exist_ok=True)
            
            download_kaggle_dataset(
                dataset["name"], 
                dataset["download_path"], 
                dataset["extract_to"]
            )
            
            # Clean up temp directory
            shutil.rmtree(dataset["download_path"])
            
        except Exception as e:
            print(f"Error downloading {dataset['name']}: {str(e)}")
            continue
    
    print("Dataset download completed!")

if __name__ == "__main__":
    main()
