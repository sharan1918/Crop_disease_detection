import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pickle
import logging

logger = logging.getLogger(__name__)

# Register custom objects to handle model loading
@tf.keras.utils.register_keras_serializable()
class GetItem(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super(GetItem, self).__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

class CropDiseasePredictor:
    def __init__(self):
        self.wheat_model = None
        self.maize_model = None
        self.wheat_classes = None
        self.maize_classes = None
        self.wheat_label_encoder = None
        self.maize_label_encoder = None
        
    def load_models(self):
        """Load trained models and label encoders"""
        logger.info("Starting model loading...")
        try:
            # Load maize model (the quick model we just trained)
            if os.path.exists('models/quick_maize_model.h5'):
                try:
                    logger.info("Loading maize model from models/quick_maize_model.h5")
                    self.maize_model = tf.keras.models.load_model('models/quick_maize_model.h5')
                    logger.info("Maize model loaded successfully")
                    
                    with open('data/maize/processed/label_encoder.pkl', 'rb') as f:
                        self.maize_label_encoder = pickle.load(f)
                    with open('data/maize/processed/classes.txt', 'r') as f:
                        self.maize_classes = [line.strip() for line in f.readlines()]
                    logger.info(f"Maize classes loaded: {self.maize_classes}")
                except Exception as e:
                    logger.error(f"Failed to load maize model: {e}")
                    self.maize_model = None
            else:
                logger.warning("Maize model file not found: models/quick_maize_model.h5")
                
        except Exception as e:
            logger.exception(f"Error loading models: {e}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for prediction"""
        logger.info(f"Preprocessing image: {image_path}")
        try:
            # Load image
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Assume it's already a numpy array
                img = image_path
            
            logger.debug(f"Original image shape: {img.shape}")
            
            # Resize
            img = cv2.resize(img, target_size)
            logger.debug(f"Resized image to: {target_size}")
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            logger.info(f"Image preprocessed successfully, final shape: {img.shape}")
            return img
            
        except Exception as e:
            logger.exception(f"Error preprocessing image: {e}")
            return None
    
    def predict_maize_disease(self, image_path):
        """Predict maize disease from image"""
        if self.maize_model is None:
            # Fallback prediction based on simple heuristics
            return self._fallback_prediction("maize")
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            return None
            
        # Make prediction
        prediction = self.maize_model.predict(np.expand_dims(image, axis=0))[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_class_idx])
        predicted_class = self.maize_classes[predicted_class_idx]
        
        # Determine health status
        health_status = "healthy" if predicted_class.lower() == "healthy" else "unhealthy"
        disease_name = None if health_status == "healthy" else predicted_class
        
        return {
            "crop_type": "maize",
            "health_status": health_status,
            "disease_name": disease_name,
            "confidence": confidence,
            "all_probabilities": {
                class_name: float(prob) for class_name, prob in zip(self.maize_classes, prediction)
            }
        }
    
    def _fallback_prediction(self, crop_type):
        """Fallback prediction when model fails to load"""
        import random
        
        classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
        probabilities = [random.random() for _ in classes]
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = classes[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        health_status = "healthy" if predicted_class.lower() == "healthy" else "unhealthy"
        disease_name = None if health_status == "healthy" else predicted_class
        
        return {
            "crop_type": crop_type,
            "health_status": health_status,
            "disease_name": disease_name,
            "confidence": confidence,
            "all_probabilities": {
                class_name: float(prob) for class_name, prob in zip(classes, probabilities)
            },
            "note": "Fallback prediction - model not loaded"
        }
    
    def predict_maize(self, image_path):
        """Predict maize disease"""
        logger.info(f"Predicting maize disease for: {image_path}")
        if self.maize_model is None:
            logger.error("Maize model not loaded, cannot predict")
            return {"error": "Maize model not loaded"}
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            logger.error("Failed to preprocess image")
            return {"error": "Failed to preprocess image"}
        
        # Make prediction
        logger.info("Running model prediction...")
        predictions = self.maize_model.predict(img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = self.maize_classes[predicted_class_idx]
        
        logger.info(f"Prediction: {predicted_class} with confidence {confidence:.4f}")
        
        # Determine if healthy or unhealthy
        is_healthy = "healthy" in predicted_class.lower()
        disease_name = predicted_class if not is_healthy else None
        
        # Get all probabilities
        all_probabilities = {}
        for i, class_name in enumerate(self.maize_classes):
            all_probabilities[class_name] = float(predictions[0][i])
        
        result = {
            "crop_type": "maize",
            "health_status": "healthy" if is_healthy else "unhealthy",
            "disease_name": disease_name,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }
        logger.info(f"Prediction result: {result}")
        return result
    
    def predict(self, image_path, crop_type=None):
        """Predict disease for given image"""
        logger.info(f"Predict called with crop_type={crop_type}")
        # Load models if not already loaded
        if self.wheat_model is None and self.maize_model is None:
            logger.info("Models not loaded, loading now...")
            self.load_models()
        
        if crop_type:
            if crop_type.lower() == "wheat":
                logger.info("Using wheat prediction")
                return self.predict_wheat(image_path)
            elif crop_type.lower() in ["maize", "corn"]:
                logger.info("Using maize prediction")
                return self.predict_maize(image_path)
            else:
                logger.error(f"Unsupported crop type: {crop_type}")
                return {"error": f"Unsupported crop type: {crop_type}"}
        else:
            logger.info("No crop_type specified, trying maize model")
            # Default to maize if no crop type specified
            return self.predict_maize(image_path)

def test_prediction():
    """Test the prediction system"""
    predictor = CropDiseasePredictor()
    predictor.load_models()
    
    # Test with sample images (you'll need to provide actual image paths)
    test_images = [
        # Add paths to test images here
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            result = predictor.predict(img_path)
            print(f"Image: {img_path}")
            print(f"Result: {result}")
            print("-" * 50)

if __name__ == "__main__":
    test_prediction()
