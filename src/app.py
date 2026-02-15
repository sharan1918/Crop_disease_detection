from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename
import uuid
from inference import CropDiseasePredictor

app = Flask(__name__, template_folder='../templates')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
predictor = CropDiseasePredictor()
predictor.load_models()

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """Serve the dashboard"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "Crop Disease Classification API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict crop disease from image",
            "/predict_url": "POST - Predict from image URL", 
            "/health": "GET - Check API health",
            "/models/info": "GET - Model information"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "wheat": predictor.wheat_model is not None,
            "maize": predictor.maize_model is not None
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from uploaded image"""
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use: png, jpg, jpeg, gif"}), 400
    
    # Get crop type from form data (optional)
    crop_type = request.form.get('crop_type', None)
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        result = predictor.predict(filepath, crop_type)
        
        # Clean up temporary file
        os.remove(filepath)
        
        # Check for prediction errors
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_url', methods=['POST'])
def predict_from_url():
    """Predict crop disease from image URL"""
    
    data = request.get_json()
    
    if not data or 'image_url' not in data:
        return jsonify({"error": "image_url is required"}), 400
    
    image_url = data['image_url']
    crop_type = data.get('crop_type', None)
    
    try:
        import requests
        from io import BytesIO
        
        # Download image from URL
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name
        
        # Make prediction
        result = predictor.predict(tmp_filepath, crop_type)
        
        # Clean up temporary file
        os.remove(tmp_filepath)
        
        # Check for prediction errors
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        # Clean up file if it exists
        if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/models/info')
def models_info():
    """Get information about available models"""
    info = {
        "wheat": {
            "loaded": predictor.wheat_model is not None,
            "classes": predictor.wheat_classes if predictor.wheat_classes else []
        },
        "maize": {
            "loaded": predictor.maize_model is not None,
            "classes": predictor.maize_classes if predictor.maize_classes else []
        }
    }
    
    return jsonify(info)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Crop Disease Classification API...")
    print("Available endpoints:")
    print("  POST /predict - Upload image file")
    print("  POST /predict_url - Predict from image URL")
    print("  GET /health - Health check")
    print("  GET /models/info - Model information")
    print("\nAPI will be available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
