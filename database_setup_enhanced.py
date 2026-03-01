#!/usr/bin/env python3
"""
Enhanced Database Setup for Crop Disease Detection and Yield Prediction
Indian Agriculture Professional Dashboard
"""

import sqlite3
import hashlib
import os
from pathlib import Path

def setup_enhanced_database():
    """Create comprehensive database for Indian agriculture dashboard"""
    
    db_path = Path(r"C:\Users\Vijay\OneDrive\Desktop\Crop_disease_detection\database")
    db_path.mkdir(exist_ok=True)
    
    db_file = db_path / "indian_agriculture.db"
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Users table with Indian context
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            phone VARCHAR(20) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(100),
            farm_name VARCHAR(100),
            state VARCHAR(50),
            district VARCHAR(50),
            village VARCHAR(50),
            land_size_acres FLOAT,
            preferred_language VARCHAR(10) DEFAULT 'en',
            theme_preference VARCHAR(10) DEFAULT 'light',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Crop disease prediction history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            crop_type VARCHAR(50) NOT NULL,
            image_path VARCHAR(255),
            prediction_result VARCHAR(100),
            confidence_score FLOAT,
            disease_name VARCHAR(100),
            severity_level VARCHAR(20),
            treatment_recommendation TEXT,
            all_probabilities TEXT,
            gps_lat FLOAT,
            gps_lng FLOAT,
            weather_condition VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Yield prediction table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS yield_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            crop_type VARCHAR(50) NOT NULL,
            field_name VARCHAR(100),
            area_acres FLOAT,
            soil_type VARCHAR(50),
            soil_ph FLOAT,
            soil_nitrogen FLOAT,
            soil_phosphorus FLOAT,
            soil_potassium FLOAT,
            irrigation_type VARCHAR(50),
            rainfall_mm FLOAT,
            temperature_avg FLOAT,
            humidity_avg FLOAT,
            fertilizer_n_kg FLOAT,
            fertilizer_p_kg FLOAT,
            fertilizer_k_kg FLOAT,
            planting_date DATE,
            expected_harvest_date DATE,
            predicted_yield_tons FLOAT,
            confidence_score FLOAT,
            market_price_per_ton FLOAT,
            estimated_revenue FLOAT,
            risk_factors TEXT,
            recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Indian crop calendar
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crop_calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name VARCHAR(50) NOT NULL,
            state VARCHAR(50) NOT NULL,
            sowing_month_start INTEGER,
            sowing_month_end INTEGER,
            harvesting_month_start INTEGER,
            harvesting_month_end INTEGER,
            growing_days INTEGER,
            water_requirement_mm FLOAT,
            optimal_temperature_min FLOAT,
            optimal_temperature_max FLOAT,
            soil_ph_min FLOAT,
            soil_ph_max FLOAT,
            is_rabi_crop BOOLEAN DEFAULT 0,
            is_kharif_crop BOOLEAN DEFAULT 0,
            is_zaid_crop BOOLEAN DEFAULT 0
        )
    ''')
    
    # Market prices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name VARCHAR(50) NOT NULL,
            state VARCHAR(50) NOT NULL,
            market_name VARCHAR(100),
            price_per_quintal FLOAT,
            price_date DATE,
            price_trend VARCHAR(20),
            quality_grade VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Weather data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state VARCHAR(50) NOT NULL,
            district VARCHAR(50) NOT NULL,
            date DATE NOT NULL,
            temperature_max FLOAT,
            temperature_min FLOAT,
            rainfall_mm FLOAT,
            humidity_percent FLOAT,
            wind_speed_kmph FLOAT,
            weather_condition VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token VARCHAR(255) UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert Indian crop calendar data
    indian_crops = [
        # Kharif crops
        ('Rice', 'Punjab', 6, 7, 10, 11, 120, 1200, 20, 35, 5.5, 7.0, 0, 1, 0),
        ('Maize', 'Maharashtra', 6, 7, 9, 10, 90, 800, 15, 30, 5.5, 7.5, 0, 1, 0),
        ('Cotton', 'Gujarat', 5, 6, 10, 12, 160, 1000, 20, 35, 6.0, 8.0, 0, 1, 0),
        
        # Rabi crops
        ('Wheat', 'Uttar Pradesh', 10, 11, 3, 4, 120, 600, 10, 25, 6.0, 7.5, 1, 0, 0),
        ('Mustard', 'Rajasthan', 10, 11, 2, 3, 110, 400, 8, 25, 6.5, 8.0, 1, 0, 0),
        
        # Zaid crops
        ('Watermelon', 'Maharashtra', 2, 3, 5, 6, 80, 600, 20, 35, 6.0, 7.5, 0, 0, 1),
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO crop_calendar 
        (crop_name, state, sowing_month_start, sowing_month_end, 
         harvesting_month_start, harvesting_month_end, growing_days,
         water_requirement_mm, optimal_temperature_min, optimal_temperature_max,
         soil_ph_min, soil_ph_max, is_rabi_crop, is_kharif_crop, is_zaid_crop)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', indian_crops)
    
    # Insert sample market prices
    market_prices = [
        ('Wheat', 'Punjab', 'Mandi', 2200, '2024-01-15', 'stable', 'A'),
        ('Rice', 'Andhra Pradesh', 'Kurnool', 1800, '2024-01-15', 'increasing', 'A'),
        ('Maize', 'Maharashtra', 'Nagpur', 1600, '2024-01-15', 'stable', 'B'),
        ('Cotton', 'Gujarat', 'Ahmedabad', 6500, '2024-01-15', 'decreasing', 'A'),
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO market_prices 
        (crop_name, state, market_name, price_per_quintal, price_date, price_trend, quality_grade)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', market_prices)
    
    # Admin user (Indian farmer profile)
    default_password = "admin123"
    password_hash = hashlib.sha256(default_password.encode()).hexdigest()
    
    cursor.execute('''
        INSERT OR IGNORE INTO users 
        (username, email, phone, password_hash, full_name, farm_name, 
         state, district, village, land_size_acres)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', ("admin", "admin@krishi.gov.in", "+919876543210", password_hash,
          "Rajesh Kumar", "Shanti Farms", "Punjab", "Ludhiana", "Doraha", 15.5))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Enhanced Indian Agriculture Database created at: {db_file}")
    print("🌾 Default Indian Farmer Profile:")
    print("   Username: admin")
    print("   Email: admin@krishi.gov.in")
    print("   Phone: +919876543210")
    print("   Password: admin123")
    print("   Farm: Shanti Farms, Punjab (15.5 acres)")

if __name__ == "__main__":
    setup_enhanced_database()
