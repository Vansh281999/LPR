import sqlite3
import logging
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='society_vehicles.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
        
    def init_database(self):
        """Initialize database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vehicle owners table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_owners (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_name TEXT NOT NULL,
                license_plate TEXT UNIQUE NOT NULL,
                apartment_number TEXT,
                contact_phone TEXT,
                whatsapp_number TEXT,
                telegram_chat_id TEXT,
                email TEXT,
                is_authorized BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Vehicle logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT,
                detected_plate TEXT,
                confidence REAL,
                accuracy REAL,
                is_authorized BOOLEAN,
                image_path TEXT,
                weather_conditions TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (license_plate) REFERENCES vehicle_owners (license_plate)
            )
        ''')
        
        # System performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_tests INTEGER,
                successful_detections INTEGER,
                average_accuracy REAL,
                average_processing_time REAL,
                weather_conditions TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("Database initialized successfully")
        
    def add_vehicle_owner(self, owner_name, license_plate, apartment_number, 
                         contact_phone, whatsapp_number=None, telegram_chat_id=None, email=None):
        """Add a new vehicle owner to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO vehicle_owners 
                (owner_name, license_plate, apartment_number, contact_phone, whatsapp_number, telegram_chat_id, email)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (owner_name, license_plate, apartment_number, contact_phone, whatsapp_number, telegram_chat_id, email))
            
            conn.commit()
            self.logger.info(f"Added vehicle owner: {owner_name} - {license_plate}")
            return True
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"License plate already exists: {license_plate}")
            return False
        finally:
            conn.close()
            
    def verify_license_plate(self, license_plate):
        """Verify if a license plate is authorized"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT owner_name, apartment_number, contact_phone, whatsapp_number, telegram_chat_id, email
            FROM vehicle_owners 
            WHERE license_plate = ? AND is_authorized = 1
        ''', (license_plate,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'owner_name': result[0],
                'apartment_number': result[1],
                'contact_phone': result[2],
                'whatsapp_number': result[3],
                'telegram_chat_id': result[4],
                'email': result[5],
                'is_authorized': True
            }
        return {'is_authorized': False}
        
    def log_vehicle_detection(self, license_plate, detected_plate, confidence, 
                            accuracy, is_authorized, image_path, weather_conditions, processing_time):
        """Log vehicle detection event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO vehicle_logs 
            (license_plate, detected_plate, confidence, accuracy, is_authorized, image_path, weather_conditions, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (license_plate, detected_plate, confidence, accuracy, is_authorized, image_path, weather_conditions, processing_time))
        
        conn.commit()
        conn.close()
        
    def get_performance_stats(self):
        """Get system performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_detections,
                AVG(accuracy) as avg_accuracy,
                AVG(processing_time) as avg_processing_time,
                COUNT(CASE WHEN is_authorized = 1 THEN 1 END) as authorized_count,
                COUNT(CASE WHEN is_authorized = 0 THEN 1 END) as unauthorized_count
            FROM vehicle_logs
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_detections': result[0],
            'average_accuracy': round(result[1] or 0, 2),
            'average_processing_time': round(result[2] or 0, 2),
            'authorized_vehicles': result[3],
            'unauthorized_vehicles': result[4]
        }