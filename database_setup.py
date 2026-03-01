#!/usr/bin/env python3
"""
Database setup for Crop Disease Detection Dashboard
SQLite database with user authentication and history tracking
"""

import sqlite3
import hashlib
import os
from pathlib import Path

def setup_database():
    """Create SQLite database with user authentication and prediction history"""
    
    db_path = Path(r"C:\Users\Vijay\OneDrive\Desktop\Crop_disease_detection\database")
    db_path.mkdir(exist_ok=True)
    
    db_file = db_path / "crop_disease.db"
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            phone VARCHAR(20) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Prediction history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path VARCHAR(255),
            crop_type VARCHAR(50),
            prediction_result VARCHAR(100),
            confidence_score FLOAT,
            disease_name VARCHAR(100),
            all_probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # User sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token VARCHAR(255) UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Admin user (default)
    default_password = "admin123"
    password_hash = hashlib.sha256(default_password.encode()).hexdigest()
    
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, phone, password_hash)
        VALUES (?, ?, ?, ?)
    ''', ("admin", "admin@cropdisease.com", "+1234567890", password_hash))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created at: {db_file}")
    print("📊 Default admin user:")
    print("   Username: admin")
    print("   Email: admin@cropdisease.com")
    print("   Phone: +1234567890")
    print("   Password: admin123")

if __name__ == "__main__":
    setup_database()
