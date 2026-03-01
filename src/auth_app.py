from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import sqlite3
import hashlib
import uuid
import datetime
import os
from functools import wraps
from werkzeug.utils import secure_filename
import tempfile
from inference import CropDiseasePredictor

app = Flask(__name__, template_folder='../templates')
CORS(app)
app.secret_key = 'your-secret-key-change-in-production'

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'crop_disease.db')

# Initialize predictor
predictor = CropDiseasePredictor()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def generate_session_token():
    """Generate random session token"""
    return str(uuid.uuid4())

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'success': False, 'message': 'No token provided'}), 401
        
        token = token.replace('Bearer ', '')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if session token is valid
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.session_token = ? AND s.expires_at > datetime('now')
        ''', (token,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'success': False, 'message': 'Invalid or expired token'}), 401
        
        request.current_user = dict(user)
        return f(*args, **kwargs)
    
    return decorated_function

@app.route('/')
def index():
    """Serve Amazon dashboard"""
    return render_template('amazon_dashboard.html')

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.get_json()
        identifier = data.get('identifier', '').strip()
        password = data.get('password', '')
        
        if not identifier or not password:
            return jsonify({'success': False, 'message': 'Email/phone and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Find user by email or phone
        cursor.execute('''
            SELECT * FROM users 
            WHERE (email = ? OR phone = ?) AND is_active = 1
        ''', (identifier, identifier))
        
        user = cursor.fetchone()
        
        if not user or not verify_password(password, user['password_hash']):
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
        
        # Generate session token
        token = generate_session_token()
        expires_at = datetime.datetime.now() + datetime.timedelta(days=7)
        
        # Save session
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user['id'], token, expires_at))
        
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = datetime('now') WHERE id = ?
        ''', (user['id'],))
        
        conn.commit()
        conn.close()
        
        user_dict = dict(user)
        user_dict.pop('password_hash', None)  # Remove password hash
        
        return jsonify({
            'success': True,
            'user': user_dict,
            'token': token
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if token:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Remove session
            cursor.execute('DELETE FROM user_sessions WHERE session_token = ?', (token,))
            
            conn.commit()
            conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/validate-session', methods=['POST'])
def validate_session():
    """Validate session token"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'valid': False}), 401
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.session_token = ? AND s.expires_at > datetime('now')
        ''', (token,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'valid': False}), 401
        
        user_dict = dict(user)
        user_dict.pop('password_hash', None)
        
        return jsonify({
            'valid': True,
            'user': user_dict
        })
        
    except Exception as e:
        return jsonify({'valid': False, 'message': str(e)}), 500

@app.route('/api/user-history', methods=['GET'])
@login_required
def get_user_history():
    """Get user's prediction history"""
    try:
        user_id = request.current_user['id']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM prediction_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 50
        ''', (user_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts and parse probabilities
        history_list = []
        for item in history:
            item_dict = dict(item)
            if item_dict['all_probabilities']:
                try:
                    item_dict['all_probabilities'] = eval(item_dict['all_probabilities'])
                except:
                    item_dict['all_probabilities'] = {}
            history_list.append(item_dict)
        
        return jsonify({'history': history_list})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image prediction with authentication"""
    try:
        user_id = request.current_user['id']
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        crop_type = request.form.get('crop_type', 'maize')
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            # Make prediction
            if crop_type.lower() == 'maize':
                result = predictor.predict_maize(temp_path)
            else:
                result = predictor.predict_wheat(temp_path)
            
            # Save to database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_history 
                (user_id, image_path, crop_type, prediction_result, confidence_score, disease_name, all_probabilities)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                filename,
                crop_type,
                result['health_status'],
                result['confidence'],
                result.get('disease_name', ''),
                str(result.get('all_probabilities', {}))
            ))
            
            conn.commit()
            conn.close()
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register():
    """Handle user registration"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not all([username, email, phone, password]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('''
            SELECT id FROM users 
            WHERE email = ? OR phone = ? OR username = ?
        ''', (email, phone, username))
        
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'User already exists'}), 400
        
        # Create user
        password_hash = hash_password(password)
        
        cursor.execute('''
            INSERT INTO users (username, email, phone, password_hash)
            VALUES (?, ?, ?, ?)
        ''', (username, email, phone, password_hash))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update-profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        user_id = request.current_user['id']
        data = request.get_json()
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        
        if not all([username, email, phone]):
            return jsonify({'success': False, 'message': 'All fields required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if email/phone is taken by another user
        cursor.execute('''
            SELECT id FROM users 
            WHERE (email = ? OR phone = ? OR username = ?) AND id != ?
        ''', (email, phone, username, user_id))
        
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Email, phone, or username already taken'}), 400
        
        # Update user
        cursor.execute('''
            UPDATE users SET username = ?, email = ?, phone = ?
            WHERE id = ?
        ''', (username, email, phone, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Amazon Dashboard API is running',
        'features': ['authentication', 'database', 'predictions', 'history']
    })

@app.route('/models/info')
def models_info():
    """Get model information"""
    try:
        predictor.load_models()
        
        info = {
            'maize_model': {
                'loaded': predictor.maize_model is not None,
                'classes': predictor.maize_classes or []
            },
            'wheat_model': {
                'loaded': predictor.wheat_model is not None,
                'classes': predictor.wheat_classes or []
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure database exists
    if not os.path.exists(DB_PATH):
        print("Database not found. Please run database_setup.py first.")
    
    # Load models
    predictor.load_models()
    
    print("🌾 Amazon Dashboard Starting...")
    print("📊 Features: Authentication, Database, Predictions, History")
    print("🔗 Dashboard: http://localhost:5000")
    print("👤 Default Admin: admin / admin123")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
