from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
import threading
import logging
from werkzeug.utils import secure_filename
import uuid
from inference import CropDiseasePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

_model_load_lock = threading.Lock()
_model_load_started = False

def _load_models_background():
    try:
        logger.info("Loading models in background...")
        predictor.load_models()
        logger.info("Background model loading finished")
    except Exception as e:
        logger.error(f"Background model loading failed: {e}")

def ensure_models_loading():
    global _model_load_started
    if _model_load_started:
        return
    with _model_load_lock:
        if _model_load_started:
            return
        _model_load_started = True
        threading.Thread(target=_load_models_background, daemon=True).start()

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logger.warning("No Gemini API key found in environment")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_client = genai
        logger.info("Gemini client initialized successfully using google.generativeai")
        return _gemini_client
    except ImportError:
        logger.warning("google.generativeai not available, trying google.genai")
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully using google.genai")
            return _gemini_client
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None

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
            "/models/info": "GET - Model information",
            "/chat": "POST - Gemini chat endpoint"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    ensure_models_loading()
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
    logger.info("Received prediction request")
    ensure_models_loading()
    
    # Check if file is in request
    if 'file' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({"error": "File type not allowed. Use: png, jpg, jpeg, gif"}), 400
    
    # Get crop type from form data (optional)
    crop_type = request.form.get('crop_type', None)
    logger.info(f"Processing file: {file.filename}, crop_type: {crop_type}")
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        logger.info(f"Saved file to: {filepath}")
        
        # Make prediction
        logger.info("Starting prediction...")
        result = predictor.predict(filepath, crop_type)
        logger.info(f"Prediction result: {result}")
        
        # Clean up temporary file
        os.remove(filepath)
        logger.info(f"Cleaned up file: {filepath}")
        
        # Check for prediction errors
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_url', methods=['POST'])
def predict_from_url():
    """Predict crop disease from image URL"""
    logger.info("Received predict_url request")
    ensure_models_loading()
    
    data = request.get_json()
    
    if not data or 'image_url' not in data:
        logger.warning("No image_url provided")
        return jsonify({"error": "image_url is required"}), 400
    
    image_url = data['image_url']
    crop_type = data.get('crop_type', None)
    logger.info(f"Downloading image from: {image_url}, crop_type: {crop_type}")
    
    try:
        import requests
        from io import BytesIO
        
        # Download image from URL
        logger.info("Downloading image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        logger.info(f"Downloaded {len(response.content)} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name
        logger.info(f"Saved to temp file: {tmp_filepath}")
        
        # Make prediction
        logger.info("Starting prediction...")
        result = predictor.predict(tmp_filepath, crop_type)
        logger.info(f"Prediction result: {result}")
        
        # Clean up temporary file
        os.remove(tmp_filepath)
        
        # Check for prediction errors
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            return jsonify(result), 500
        
        return jsonify(result)
        
    except requests.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        # Clean up file if it exists
        if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/models/info')
def models_info():
    """Get information about available models"""
    ensure_models_loading()
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

@app.route('/chat', methods=['POST'])
def chat():
    ensure_models_loading()

    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    analysis_result = data.get('analysis_result', None)

    if not message:
        return jsonify({"error": "message is required"}), 400

    client = get_gemini_client()
    if client is None:
        return jsonify({"error": "Gemini API key not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in environment."}), 503

    model = os.getenv('GEMINI_MODEL') or 'gemini-2.0-flash'

    context_block = ""
    if isinstance(analysis_result, dict) and analysis_result:
        crop_type = analysis_result.get('crop_type')
        health_status = analysis_result.get('health_status')
        disease_name = analysis_result.get('disease_name')
        confidence = analysis_result.get('confidence')
        context_block = (
            "Latest classifier result (may be incomplete):\n"
            f"- crop_type: {crop_type}\n"
            f"- health_status: {health_status}\n"
            f"- disease_name: {disease_name}\n"
            f"- confidence: {confidence}\n"
        )

    system_text = (
        "You are an assistant for a crop disease classification dashboard. "
        "Answer questions about crop diseases, symptoms, prevention, and how to interpret the model's result. "
        "If the user asks for medical or unsafe chemical advice, provide general safe guidance and recommend local agricultural extension services. "
        "Be concise and practical."
    )

    prompt = f"{system_text}\n\n{context_block}\nUser question: {message}"

    try:
        logger.info(f"Sending request to Gemini model: {model}")
        
        # Try using google.generativeai first (older but stable SDK)
        if hasattr(client, 'GenerativeModel'):
            gemini_model = client.GenerativeModel(model)
            response = gemini_model.generate_content(prompt)
            text = response.text
        else:
            # Fallback to google.genai (newer unified SDK)
            response = client.models.generate_content(model=model, contents=prompt)
            text = getattr(response, 'text', None)
            if not text:
                text = str(response)
        
        logger.info("Gemini request successful")
        return jsonify({"answer": text})
    except Exception as e:
        logger.exception(f"Gemini request failed: {e}")
        return jsonify({"error": f"Gemini request failed: {str(e)}"}), 500

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
    logger.info("Starting Crop Disease Classification API...")
    logger.info("Available endpoints:")
    logger.info("  POST /predict - Upload image file")
    logger.info("  POST /predict_url - Predict from image URL")
    logger.info("  GET /health - Health check")
    logger.info("  GET /models/info - Model information")
    logger.info("API will be available at: http://localhost:5000")

    ensure_models_loading()
     
    app.run(host='0.0.0.0', port=3000, debug=True)
