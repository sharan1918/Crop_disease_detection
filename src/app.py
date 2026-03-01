from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import tempfile
import threading
import logging
import uuid
import datetime
from inference import CropDiseasePredictor
from yield_predictor import YieldPredictor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AI Intelligence Core
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini AI Engine Online")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        model = None
else:
    logger.warning("GOOGLE_API_KEY not found. Gemini AI chat will be disabled.")
    model = None

app = Flask(__name__, template_folder='../templates')
app.config['SECRET_KEY'] = 'crop-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), unique=True)
    password_hash = db.Column(db.String(128))
    full_name = db.Column(db.String(100))
    location = db.Column(db.String(100))
    otp = db.Column(db.String(6))
    otp_expiry = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    settings_data = db.Column(db.Text, default='{}')

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    crop_type = db.Column(db.String(50))
    prediction_type = db.Column(db.String(20)) # 'disease' or 'yield'
    result = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Load Models
predictor = CropDiseasePredictor()
yield_predictor = YieldPredictor()

_model_load_lock = threading.Lock()
_model_load_started = False

def _load_models_background():
    try:
        predictor.load_models()
        logger.info("Models loaded successfully in background")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Auth Routes ---
import re
import random

def is_valid_phone(phone):
    # Remove any non-digit characters for validation
    digits = re.sub(r'\D', '', str(phone))
    # Handle optional country code +91 or 0
    if len(digits) == 12 and digits.startswith('91'):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith('0'):
        digits = digits[1:]
    return len(digits) == 10 and digits[0] in '6789'

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Security Alert: All credentials are required."}), 400
    
    # Check by username or phone
    user = User.query.filter((User.username == username) | (User.phone == username)).first()
    
    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        return jsonify({
            "message": "Authenticated",
            "user": {"username": user.username, "full_name": user.full_name}
        })
    
    return jsonify({"error": "Invalid Identity or Secret Key."}), 401

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get('username')
    email = data.get('email')
    phone = data.get('phone')
    password = data.get('password')

    if not all([username, email, phone, password]):
        return jsonify({"error": "All fields are required"}), 400
    
    if not is_valid_phone(phone):
        return jsonify({"error": "Please enter a valid 10-digit mobile number"}), 400

    # Clean the phone number for storage (just 10 digits)
    clean_phone = re.sub(r'\D', '', str(phone))[-10:]

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    
    if User.query.filter_by(phone=clean_phone).first():
        return jsonify({"error": "Phone number already registered"}), 400
    
    try:
        user = User(
            username=username,
            email=email,
            phone=clean_phone,
            password_hash=generate_password_hash(password),
            full_name=data.get('full_name', ''),
            location=data.get('location', '')
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration Error: {e}")
        return jsonify({"error": "Failed to create account. Database may need update."}), 500

@app.route('/api/logout')
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@app.route('/api/auth/status')
def auth_status():
    if current_user.is_authenticated:
        return jsonify({
            "isLoggedIn": True,
            "user": {"username": current_user.username, "full_name": current_user.full_name}
        })
    return jsonify({"isLoggedIn": False})

# --- Main Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "maize": predictor.maize_model is not None,
            "wheat": predictor.wheat_model is not None,
            "tomato": predictor.tomato_model is not None
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    
    crop_type = request.form.get('crop_type', 'maize')
    
    # Process from memory for speed
    try:
        import numpy as np
        import cv2
        
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Predict directly using the decoded image
        result = predictor.predict(img, crop_type)
        
        # Save history in background to avoid delaying response
        if current_user.is_authenticated and "error" not in result:
            def save_history(uid, ctype, res):
                with app.app_context():
                    prediction = Prediction(
                        user_id=uid,
                        crop_type=ctype,
                        prediction_type='disease',
                        result=res
                    )
                    db.session.add(prediction)
                    db.session.commit()
            
            threading.Thread(target=save_history, args=(current_user.id, crop_type, result), daemon=True).start()
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_yield', methods=['POST'])
def predict_yield():
    data = request.get_json()
    try:
        result = yield_predictor.predict(data)
        
        # Save history if logged in
        if current_user.is_authenticated:
            # Handle multiple crops if applicable
            crops_list = data.get('crops', [])
            crop_name = ", ".join(crops_list) if isinstance(crops_list, list) else str(crops_list)
            
            prediction = Prediction(
                user_id=current_user.id,
                crop_type=crop_name,
                prediction_type='yield',
                result=result
            )
            db.session.add(prediction)
            db.session.commit()
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history')
@login_required
def get_history():
    history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(20).all()
    return jsonify([{
        "crop_type": h.crop_type,
        "type": h.prediction_type,
        "result": h.result,
        "timestamp": h.timestamp.strftime('%Y-%m-%d %H:%M')
    } for h in history])

@app.route('/api/stats')
def get_stats():
    # Public stats mockup
    return jsonify({
        "total_farmers": User.query.count() + 1500, # Mocked base + real
        "predictions_today": 45,
        "active_regions": ["Maharashtra", "Punjab", "Karnataka"]
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    context = data.get('context')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    if not model:
        # Static context-aware fallback
        fallback = "I am currently in offline mode."
        if context:
            fallback = f"I see you are analyzing {context.get('crop_type')} with {context.get('disease_name', 'potential issues')}. Please consult a local expert."
        return jsonify({"response": fallback})
    
    try:
        # Add context to prompt if available
        context_str = ""
        if context:
            status = context.get('health_status', 'unhealthy')
            disease = context.get('disease_name', 'unspecified symptoms')
            context_str = f"[CONTEXT: Farmer has scanned {context.get('crop_type')}. Status: {status}. Detected: {disease}.] "

        prompt = f"You are an expert AI Crop Doctor for AgriQuest. {context_str} Answer this farmer's query concisely and scientifically: {user_message}"
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return jsonify({"error": "Intelligence Core failure. Please try later."}), 500

@app.route('/api/experts')
def get_experts():
    # In a real app, this would query a DB based on lat/lon
    # Here we return a list of experts with mock distances
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    # Mock data
    experts = [
        {
            "name": "Dr. Rajesh Kumar",
            "specialty": "Soil Pathologist",
            "rating": "4.9",
            "experience": "12Y",
            "image": "https://ui-avatars.com/api/?name=Rajesh+Kumar&background=0D8ABC&color=fff",
            "distance": "1.2 km" if lat else "Nearby"
        },
        {
            "name": "Anjali Mehra",
            "specialty": "Maize Specialist",
            "rating": "4.8",
            "experience": "8Y",
            "image": "https://ui-avatars.com/api/?name=Anjali+Mehra&background=E10000&color=fff",
            "distance": "3.5 km" if lat else "Nearby"
        },
        {
            "name": "Prof. S. Swaminathan",
            "specialty": "Cereal Pathologist",
            "rating": "5.0",
            "experience": "25Y",
            "image": "https://ui-avatars.com/api/?name=S+Swaminathan&background=00A36C&color=fff",
            "distance": "5.1 km" if lat else "Nearby"
        }
    ]
    return jsonify(experts)

@app.route('/api/models/info')
def models_info():
    return jsonify({
        "maize": {
            "loaded": predictor.maize_model is not None,
            "classes": predictor.maize_classes if predictor.maize_classes else ["Blight", "Rust", "Spot", "Healthy"]
        },
        "wheat": {
            "loaded": predictor.wheat_model is not None,
            "classes": predictor.wheat_classes if predictor.wheat_classes else ["Smut", "Yellow Rust", "Healthy"]
        },
        "tomato": {
            "loaded": predictor.tomato_model is not None,
            "classes": predictor.tomato_classes if predictor.tomato_classes else ["Leaf Blight", "Bacterial Spot", "Healthy"]
        }
    })

import json

@app.route('/api/settings', methods=['GET'])
@login_required
def get_settings():
    settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if settings:
        return jsonify({"settings": json.loads(settings.settings_data)})
    return jsonify({"settings": {}})

@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    data = request.get_json()
    settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    
    if settings:
        # Merge new data with existing
        existing = json.loads(settings.settings_data)
        existing.update(data)
        settings.settings_data = json.dumps(existing)
    else:
        settings = UserSettings(user_id=current_user.id, settings_data=json.dumps(data))
        db.session.add(settings)
        
    db.session.commit()
    return jsonify({"message": "Settings updated successfully", "settings": json.loads(settings.settings_data)})

# Initialize DB and run
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    # Start background model loading
    threading.Thread(target=_load_models_background, daemon=True).start()
    
    app.run(host='0.0.0.0', port=3000, debug=True)
