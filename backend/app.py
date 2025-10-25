import os
import cv2
import time
import logging
import argparse
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from lpr_system import LicensePlateRecognition
from database import DatabaseManager
from notification_service import NotificationService

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lpr_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
lpr_system = LicensePlateRecognition()
db_manager = DatabaseManager()
notification_service = NotificationService(
    twilio_account_sid=os.environ.get('TWILIO_ACCOUNT_SID'),
    twilio_auth_token=os.environ.get('TWILIO_AUTH_TOKEN'),
    telegram_bot_token=os.environ.get('TELEGRAM_BOT_TOKEN')
)

@app.route('/', methods=['GET'])
def index():
    """Root UI for Smart LPR System"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'online',
        'message': 'License Plate Recognition API is running',
        'endpoints': {
            'process_image': '/api/process-image',
            'register_vehicle': '/api/register-vehicle',
            'stats': '/api/performance-stats'
        }
    })

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process an image for license plate recognition"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    # Get image file and weather conditions
    image_file = request.files['image']
    weather_conditions = request.form.get('weather_conditions', 'normal')
    enhance_for_weather = weather_conditions in ['foggy', 'misty', 'rainy']
    
    # Save image temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"temp_images/{timestamp}.jpg"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image_file.save(image_path)
    
    # Process image
    start_time = time.time()
    image, plate_img, plate_contour = lpr_system.detect_license_plate(
        image_path, 
        enhance_for_weather=enhance_for_weather
    )
    
    if plate_img is None:
        processing_time = time.time() - start_time
        return jsonify({
            'error': 'No license plate detected',
            'processing_time': processing_time
        }), 404
        
    # Extract text from plate
    plate_text, confidence = lpr_system.extract_text_from_plate(plate_img)
    processing_time = time.time() - start_time
    
    if not plate_text:
        return jsonify({
            'error': 'Could not read license plate text',
            'processing_time': processing_time
        }), 404
        
    # Verify if plate is authorized
    owner_info = db_manager.verify_license_plate(plate_text)
    is_authorized = owner_info.get('is_authorized', False)
    
    # Calculate accuracy (if we have ground truth)
    accuracy = None
    if 'expected_plate' in request.form:
        expected_plate = request.form['expected_plate']
        accuracy = lpr_system.calculate_accuracy(plate_text, expected_plate)
        
    # Log detection
    db_manager.log_vehicle_detection(
        license_plate=plate_text if is_authorized else None,
        detected_plate=plate_text,
        confidence=confidence,
        accuracy=accuracy,
        is_authorized=is_authorized,
        image_path=image_path,
        weather_conditions=weather_conditions,
        processing_time=processing_time
    )
    
    # Send notification if authorized
    if is_authorized:
        notification_service.notify_vehicle_owner(
            owner_info,
            "authorized_entry",
            plate_text,
            accuracy
        )
    
    # Return results
    return jsonify({
        'plate_text': plate_text,
        'confidence': confidence,
        'accuracy': accuracy,
        'is_authorized': is_authorized,
        'processing_time': processing_time,
        'owner_info': owner_info if is_authorized else None
    })

@app.route('/api/register-vehicle', methods=['POST'])
def register_vehicle():
    """Register a new vehicle owner"""
    data = request.json
    
    required_fields = ['owner_name', 'license_plate', 'apartment_number', 'contact_phone']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
            
    success = db_manager.add_vehicle_owner(
        owner_name=data['owner_name'],
        license_plate=data['license_plate'],
        apartment_number=data['apartment_number'],
        contact_phone=data['contact_phone'],
        whatsapp_number=data.get('whatsapp_number'),
        telegram_chat_id=data.get('telegram_chat_id'),
        email=data.get('email')
    )
    
    if success:
        return jsonify({'message': 'Vehicle registered successfully'}), 201
    else:
        return jsonify({'error': 'License plate already exists'}), 409

@app.route('/api/performance-stats', methods=['GET'])
def get_performance_stats():
    """Get system performance statistics"""
    stats = db_manager.get_performance_stats()
    return jsonify(stats)

def main():
    """Run the application"""
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Create temp directory for images
    os.makedirs('temp_images', exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()