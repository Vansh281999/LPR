import os
import io
import cv2
import json
import re
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
from werkzeug.utils import secure_filename
import base64

# Import detection functions
from lpr_detector_v2 import detect_and_recognize_plate, init_detector

# Import notification functions
from lpr import send_email_notification, send_whatsapp_message, load_config

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Database path
DB_PATH = 'society_vehicles.db'


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings (pure Python)."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # use two-row DP for memory efficiency
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def fuzzy_match_plate(candidate: str, max_distance: int = 3):
    """Try to match a recognized plate to a registered plate using fuzzy matching.
    Returns (matched_plate, distance) or (None, None).
    """
    if not candidate:
        return None, None
    cand = re.sub(r'[^A-Z0-9]', '', candidate.upper())
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('SELECT plate_number FROM owners')
        rows = [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    best = (None, 999)
    for reg in rows:
        reg_clean = re.sub(r'[^A-Z0-9]', '', str(reg).upper())
        d = levenshtein(cand, reg_clean)
        if d < best[1]:
            best = (reg_clean, d)

    if best[0] is not None and best[1] <= max_distance:
        return best[0], best[1]
    return None, None

def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Owners registry (plate -> email/whatsapp)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS owners (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE,
            owner_name TEXT,
            email TEXT,
            phone TEXT,
            registered_at TEXT
        )
        """
    )
    
    # Detections log
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            image_path TEXT,
            detected_at TEXT NOT NULL
        )
        """
    )
    
    # Notifications log
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            channel TEXT,
            status TEXT,
            details TEXT,
            sent_at TEXT NOT NULL,
            FOREIGN KEY(detection_id) REFERENCES detections(id)
        )
        """
    )
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")
def ensure_schema():
    """Ensure database schema is up-to-date; migrate legacy owners table if needed."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Check owners table columns
        cur.execute("PRAGMA table_info(owners)")
        columns = [row[1] for row in cur.fetchall()]

        if 'id' not in columns:
            print("Legacy owners table detected (missing id). Migrating...")
            # Create a new table with correct schema
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS owners_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE,
                    owner_name TEXT,
                    email TEXT,
                    phone TEXT,
                    registered_at TEXT
                )
                """
            )

            # Read existing rows from legacy table
            cur.execute("SELECT * FROM owners")
            rows = cur.fetchall()

            # Determine available legacy columns
            legacy_cols = set(columns)

            # Copy row-by-row, mapping legacy fields to new schema
            for row in rows:
                plate_number = row['plate_number'] if 'plate_number' in legacy_cols else None
                if not plate_number:
                    # Skip rows without plate number
                    continue
                owner_name = row['owner_name'] if 'owner_name' in legacy_cols else ''
                email = row['email'] if 'email' in legacy_cols else ''
                # Prefer 'phone' if present; else map from 'whatsapp' if available
                phone = row['phone'] if 'phone' in legacy_cols else (row['whatsapp'] if 'whatsapp' in legacy_cols else '')
                registered_at = row['registered_at'] if 'registered_at' in legacy_cols else datetime.utcnow().isoformat()

                cur.execute(
                    """
                    INSERT INTO owners_new (plate_number, owner_name, email, phone, registered_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (plate_number, owner_name, email, phone, registered_at)
                )

            # Replace old table
            cur.execute("DROP TABLE owners")
            cur.execute("ALTER TABLE owners_new RENAME TO owners")
            conn.commit()
            print("Owners table migration completed.")

    except Exception as e:
        print(f"Schema check/migration failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/registry', methods=['GET'])
def get_registry():
    """Get all registered vehicles"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, plate_number, owner_name, email, phone FROM owners ORDER BY plate_number")
    vehicles = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return jsonify(vehicles)

@app.route('/api/registry', methods=['POST'])
def add_vehicle():
    """Add a new vehicle to registry"""
    data = request.json
    
    if not data or not data.get('plate_number') or not data.get('owner_name'):
        return jsonify({'error': 'Plate number and owner name are required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO owners (plate_number, owner_name, email, phone, registered_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                data.get('plate_number'),
                data.get('owner_name'),
                data.get('email', ''),
                data.get('phone', ''),
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        vehicle_id = cursor.lastrowid
        conn.close()
        
        return jsonify({'id': vehicle_id, 'message': 'Vehicle added successfully'}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Vehicle with this plate number already exists'}), 409
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/registry/<int:vehicle_id>', methods=['PUT'])
def update_vehicle(vehicle_id):
    """Update a vehicle in registry"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if vehicle exists
    cursor.execute("SELECT id FROM owners WHERE id = ?", (vehicle_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Vehicle not found'}), 404
    
    try:
        cursor.execute(
            """
            UPDATE owners
            SET plate_number = ?, owner_name = ?, email = ?, phone = ?
            WHERE id = ?
            """,
            (
                data.get('plate_number'),
                data.get('owner_name'),
                data.get('email', ''),
                data.get('phone', ''),
                vehicle_id
            )
        )
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Vehicle updated successfully'})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Vehicle with this plate number already exists'}), 409
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/registry/<int:vehicle_id>', methods=['DELETE'])
def delete_vehicle(vehicle_id):
    """Delete a vehicle from registry"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if vehicle exists
    cursor.execute("SELECT id FROM owners WHERE id = ?", (vehicle_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Vehicle not found'}), 404
    
    try:
        cursor.execute("DELETE FROM owners WHERE id = ?", (vehicle_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Vehicle deleted successfully'})
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get detections today
    today = datetime.utcnow().date().isoformat()
    cursor.execute("SELECT COUNT(*) FROM detections WHERE DATE(detected_at) = ?", (today,))
    detections_today = cursor.fetchone()[0]
    
    # Get total registered vehicles
    cursor.execute("SELECT COUNT(*) FROM owners")
    registered = cursor.fetchone()[0]
    
    # Get notifications count
    cursor.execute("SELECT COUNT(*) FROM notifications WHERE DATE(sent_at) = ?", (today,))
    notifications = cursor.fetchone()[0]
    
    # Calculate accuracy (mock for now - could be based on manual verification table)
    accuracy = 94.7
    
    conn.close()
    
    return jsonify({
        'accuracy': accuracy,
        'detections_today': detections_today,
        'registered': registered,
        'notifications': notifications
    })

@app.route('/api/detect', methods=['POST'])
def detect_plate():
    """Detect and recognize license plate from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image from upload
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Save original image
            filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(filepath, img)
            
            # Detect and recognize plate
            result = detect_and_recognize_plate(img)
            
            if result['success']:
                plate_number = result['plate_number']
                
                # Save to database
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO detections (plate_number, image_path, detected_at) VALUES (?, ?, ?)",
                    (plate_number, filepath, datetime.utcnow().isoformat())
                )
                detection_id = cursor.lastrowid
                
                # Check if vehicle is registered
                cursor.execute("SELECT email, whatsapp FROM owners WHERE plate_number = ?", (plate_number,))
                owner = cursor.fetchone()
                
                registered = owner is not None
                
                # Send notifications if vehicle is registered
                if registered and owner:
                    email, whatsapp = owner
                    
                    # Load config for notification settings
                    cfg = load_config()
                    
                    # Prepare notification content
                    subject = f"LPR Detection: {plate_number}"
                    body = (
                        f"A license plate was detected.\n\n"
                        f"Plate: {plate_number}\n"
                        f"Confidence: {result.get('confidence', 0.95)*100:.1f}%\n"
                        f"Status: Registered\n"
                        f"Time: {datetime.now().strftime('%I:%M:%S %p')}\n"
                        f"Detection ID: {detection_id}"
                    )
                    
                    # Send email notification
                    if email:
                        emails = [e.strip() for e in str(email).split(',') if e.strip()]
                        if emails:
                            status, details = send_email_notification(subject, body, emails, cfg)
                            cursor.execute(
                                "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
                                (detection_id, 'email', status, details, datetime.utcnow().isoformat())
                            )
                    
                    # Send WhatsApp notification
                    if whatsapp:
                        whatsapps = [w.strip() for w in str(whatsapp).split(',') if w.strip()]
                        for wa in whatsapps:
                            status, details = send_whatsapp_message(body, wa, cfg)
                            cursor.execute(
                                "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
                                (detection_id, 'whatsapp', status, details, datetime.utcnow().isoformat())
                            )
                
                # Convert annotated image to base64
                _, buffer = cv2.imencode('.jpg', result['annotated_image'])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                conn.commit()
                conn.close()
                
                return jsonify({
                    'success': True,
                    'plate_number': plate_number,
                    'confidence': result.get('confidence', 0.95),
                    'registered': registered,
                    'detection_id': detection_id,
                    'image': f'data:image/jpeg;base64,{img_base64}',
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'No plate detected')
                })
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent detections"""
    limit = request.args.get('limit', 50, type=int)
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.id, d.plate_number, d.detected_at, d.image_path,
               CASE WHEN o.plate_number IS NOT NULL THEN 1 ELSE 0 END as registered
        FROM detections d
        LEFT JOIN owners o ON d.plate_number = o.plate_number
        ORDER BY d.detected_at DESC
        LIMIT ?
    """, (limit,))
    
    detections = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(detections)

@app.route('/api/registry_legacy', methods=['GET'])
def get_registry_legacy():
    """Get all registered vehicles"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if registered_at column exists
    cursor.execute("PRAGMA table_info(owners)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'registered_at' in columns:
        cursor.execute("SELECT plate_number, email, whatsapp, registered_at FROM owners ORDER BY registered_at DESC")
    else:
        # Fallback if registered_at doesn't exist
        cursor.execute("SELECT plate_number, email, whatsapp FROM owners")
    
    registry = []
    for row in cursor.fetchall():
        data = dict(row)
        if 'registered_at' not in data:
            data['registered_at'] = datetime.utcnow().isoformat()  # Add default value
        registry.append(data)
    
    conn.close()
    
    return jsonify(registry)

@app.route('/api/registry_legacy', methods=['POST'])
def register_vehicle_legacy():
    """Register a new vehicle"""
    data = request.get_json()
    
    plate_number = data.get('plate_number', '').strip().upper()
    email = data.get('email', '').strip()
    whatsapp = data.get('whatsapp', '').strip()
    
    if not plate_number:
        return jsonify({'error': 'Plate number is required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO owners (plate_number, email, whatsapp, registered_at) VALUES (?, ?, ?, ?)",
            (plate_number, email, whatsapp, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Vehicle {plate_number} registered successfully'
        })
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Vehicle already registered'}), 400
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get recent notifications"""
    limit = request.args.get('limit', 50, type=int)
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT n.id, n.detection_id, n.channel, n.status, n.details, n.sent_at,
               d.plate_number
        FROM notifications n
        JOIN detections d ON n.detection_id = d.id
        ORDER BY n.sent_at DESC
        LIMIT ?
    """, (limit,))
    
    notifications = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(notifications)

@app.route('/api/camera_detect', methods=['POST'])
def camera_detect():
    """Process image from camera feed"""
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Decode base64 image
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save image to file
        timestamp = datetime.now().timestamp()
        filename = f"camera_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        
        # Detect and recognize license plate
        results = detect_and_recognize_plate(img)
        
        # Initialize notification flags
        notification_sent = False
        gate_opening = False
        
        if results and results['success']:
            plate_text = results['plate_number']
            
            # Save detection to database
            conn = get_db()
            cursor = conn.cursor()
            
            # Insert detection
            cursor.execute(
                "INSERT INTO detections (plate_number, image_path, detected_at) VALUES (?, ?, ?)",
                (plate_text, filepath, datetime.utcnow().isoformat())
            )
            detection_id = cursor.lastrowid
            
            # Check if vehicle is registered
            cursor.execute("SELECT id, owner_name, email, phone FROM owners WHERE plate_number = ?", (plate_text,))
            owner = cursor.fetchone()

            auto_corrected = False
            corrected_from = None

            if not owner:
                matched_plate, dist = fuzzy_match_plate(plate_text, max_distance=4)
                if matched_plate:
                    corrected_from = plate_text
                    plate_text = matched_plate
                    auto_corrected = True
                    cursor.execute("SELECT id, owner_name, email, phone FROM owners WHERE plate_number = ?", (plate_text,))
                    owner = cursor.fetchone()

            if owner:
                # Vehicle is registered, send notification
                gate_opening = True
                
                # Get owner details
                owner_id = owner['id']
                owner_name = owner['owner_name']
                email = owner['email']
                phone = owner['phone']
                
                # Send email notification if email is provided
                if email:
                    try:
                        config = load_config()
                        subject = f"Vehicle Detection: {plate_text}"
                        body = f"Your vehicle with license plate {plate_text} was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
                        status, details = send_email_notification(subject, body, [email], config)
                        
                        # Log notification
                        cursor.execute(
                            "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
                            (detection_id, 'email', status, details, datetime.utcnow().isoformat())
                        )
                        notification_sent = True
                    except Exception as e:
                        print(f"Email notification error: {str(e)}")
                        cursor.execute(
                            "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
                            (detection_id, 'email', 'failed', str(e), datetime.utcnow().isoformat())
                        )
                
                # Send WhatsApp notification if phone is provided
                if phone:
                    try:
                        config = load_config()
                        message = f"Your vehicle with license plate {plate_text} was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
                        status, details = send_whatsapp_message(message, phone, config)
                        
                        # Log notification
                        cursor.execute(
                            "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
                            (detection_id, 'whatsapp', status, details, datetime.utcnow().isoformat())
                        )
                        notification_sent = True
                    except Exception as e:
                        print(f"WhatsApp notification error: {str(e)}")
                        cursor.execute(
                            "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
                            (detection_id, 'whatsapp', 'failed', str(e), datetime.utcnow().isoformat())
                        )
            
            conn.commit()
            conn.close()
            
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', results['annotated_image'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'plate_text': plate_text,
                'confidence': results.get('confidence', 0.95),
                'image_path': filepath,
                'registered': owner is not None,
                'notification_sent': notification_sent,
                'gate_opening': gate_opening,
                'auto_corrected': auto_corrected,
                'corrected_from': corrected_from,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No license plate detected',
                'notification_sent': False,
                'gate_opening': False
            })
    
    except Exception as e:
        print(f"Error in camera_detect: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    # Ensure schema is up-to-date (migrate legacy tables if needed)
    ensure_schema()
    
    # Initialize detector
    init_detector()
    
    print("=" * 60)
    print("Smart LPR System Starting...")
    print("=" * 60)
    print(f"Frontend: http://localhost:5000")
    print(f"API: http://localhost:5000/api")
    print("=" * 60)
    
    # Disable auto-reloader to prevent reinitializing heavy OCR models
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
