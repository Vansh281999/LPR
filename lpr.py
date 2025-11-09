import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import tensorflow.keras.backend as K

# New: DB and notification imports
import sqlite3
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# ---- Paths (edit these to match your local files) ----
CASCADE_XML = os.getenv("LPR_CASCADE_XML", "data/indian_license_plate.xml")
CAR_IMAGE = os.getenv("LPR_SAMPLE_IMAGE", "data/car.jpg")
DATASET_DIR = os.getenv("LPR_DATASET_DIR", "data/data")  # expects subfolders: train/, val/

# ---- Load cascade for detecting license plates ----
plate_cascade = cv2.CascadeClassifier(CASCADE_XML)
if plate_cascade.empty():
    raise FileNotFoundError(f"Could not load cascade classifier from '{CASCADE_XML}'.")


def detect_plate(img, text=''):
    """Detect license plate in BGR image and draw rectangle; return (annotated_img, plate_roi).
    """
    plate_img = img.copy()
    roi = img.copy()
    plate = None
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y + h, x:x + w, :]
        plate = roi[y:y + h, x:x + w, :]
        cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155), 3)
        if text != '':
            plate_img = cv2.putText(
                plate_img,
                text,
                (x - w // 2, y - h // 2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.5,
                (51, 181, 155),
                1,
                cv2.LINE_AA,
            )
    return plate_img, plate


def display(img_bgr, title=''):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


# ---- Contour utilities ----
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width, upper_width, lower_height, upper_height = dimensions

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    img_res = []
    ii = img.copy()

    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if (lower_width < intWidth < upper_width) and (lower_height < intHeight < upper_height):
            x_cntr_list.append(intX)

            char_copy = np.zeros((44, 24))
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intX + intWidth, intY + intHeight), (255), 1)

            char = cv2.subtract(255, char)

            # pad to 24x44
            char_copy[2:42, 2:22] = char
            img_res.append(char_copy)

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res = np.array([img_res[idx] for idx in indices]) if indices else np.array([])
    return img_res


def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3]

    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Binarized Plate')
    plt.axis('off')
    plt.show()

    char_list = find_contours(dimensions, img_binary_lp)
    return char_list


# ---- Model utilities ----
def f1score_np(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')


def custom_f1score(y, y_pred):
    return tf.py_function(f1score_np, (y, y_pred), tf.double)


def build_model():
    K.clear_session()
    model = Sequential()
    model.add(Conv2D(16, (22, 22), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (16, 16), activation='relu', padding='same'))
    model.add(Conv2D(64, (8, 8), activation='relu', padding='same'))
    model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=[custom_f1score],
    )
    return model


def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def show_results(model, chars):
    dic = {i: c for i, c in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    output = []
    for ch in chars:
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_probs = model.predict(img, verbose=0)
        y_ = int(np.argmax(y_probs, axis=1)[0])
        output.append(dic.get(y_, '?'))
    return ''.join(output)


# ---- Config, Database and notification utilities ----
DB_PATH = os.getenv('LPR_DB_PATH', 'society_vehicles.db')
CONFIG_PATH = os.getenv('LPR_CONFIG', 'config.json')


def load_config(path: str = CONFIG_PATH):
    cfg = {
        'smtp': {
            'host': None,
            'port': 587,
            'user': None,
            'password': None,
            'sender': None
        },
        'twilio': {
            'account_sid': None,
            'auth_token': None,
            'whatsapp_from': None
        }
    }
    
    # Load from config file first (highest priority)
    if os.path.isfile(path):
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                file_cfg = json.load(f)
                # Update with non-empty values from config file
                for section in ['smtp', 'twilio']:
                    if section in file_cfg:
                        for key, value in file_cfg[section].items():
                            if value:  # Only use non-empty values
                                cfg[section][key] = value
        except Exception as e:
            print(f"Warning: Could not load config file {path}: {e}")
    
    # Fallback to environment variables if config file values are missing
    if not cfg['smtp']['host']:
        cfg['smtp']['host'] = os.getenv('SMTP_HOST')
    if not cfg['smtp']['user']:
        cfg['smtp']['user'] = os.getenv('SMTP_USER')
    if not cfg['smtp']['password']:
        cfg['smtp']['password'] = os.getenv('SMTP_PASSWORD')
    if not cfg['smtp']['sender']:
        cfg['smtp']['sender'] = os.getenv('SMTP_FROM')
    if os.getenv('SMTP_PORT'):
        try:
            cfg['smtp']['port'] = int(os.getenv('SMTP_PORT'))
        except ValueError:
            pass
    
    if not cfg['twilio']['account_sid']:
        cfg['twilio']['account_sid'] = os.getenv('TWILIO_ACCOUNT_SID')
    if not cfg['twilio']['auth_token']:
        cfg['twilio']['auth_token'] = os.getenv('TWILIO_AUTH_TOKEN')
    if not cfg['twilio']['whatsapp_from']:
        cfg['twilio']['whatsapp_from'] = os.getenv('TWILIO_WHATSAPP_FROM')

    return cfg


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Owners registry (plate -> email/whatsapp)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS owners (
            plate_number TEXT PRIMARY KEY,
            email TEXT,
            whatsapp TEXT
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
    return conn


def register_detection(conn: sqlite3.Connection, plate_number: str, image_path):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (plate_number, image_path, detected_at) VALUES (?, ?, ?)",
        (plate_number, image_path, datetime.utcnow().isoformat(timespec='seconds') + 'Z'),
    )
    conn.commit()
    return cur.lastrowid


def send_email_notification(subject: str, body: str, recipients, cfg):
    smtp_cfg = (cfg or {}).get('smtp', {})
    host = smtp_cfg.get('host')
    port = int(smtp_cfg.get('port') or 587)
    user = smtp_cfg.get('user')
    password = smtp_cfg.get('password')
    sender = smtp_cfg.get('sender')
    
    # Handle both string and list recipients
    if isinstance(recipients, str):
        recips = [recipients.strip()]
    else:
        recips = [e.strip() for e in (recipients or []) if e and e.strip()]

    if not (host and sender and recips):
        return 'skipped', 'SMTP not configured (need host/sender and at least one recipient)'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recips)

    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(sender, recips, msg.as_string())
        return 'sent', f"Email sent to {len(recips)} recipient(s)"
    except Exception as e:
        return 'error', str(e)


def record_notification(conn: sqlite3.Connection, detection_id: int, channel: str, status: str, details: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notifications (detection_id, channel, status, details, sent_at) VALUES (?, ?, ?, ?, ?)",
        (detection_id, channel, status, details, datetime.utcnow().isoformat(timespec='seconds') + 'Z'),
    )
    conn.commit()


def lookup_recipients(conn: sqlite3.Connection, plate_number: str):
    cur = conn.cursor()
    cur.execute("SELECT email, whatsapp FROM owners WHERE plate_number = ?", (plate_number,))
    row = cur.fetchone()
    emails, whatsapps = [], []
    if row:
        email, wa = row
        if email:
            emails = [e.strip() for e in str(email).split(',') if e.strip()]
        if wa:
            whatsapps = [w.strip() for w in str(wa).split(',') if w.strip()]
    return emails, whatsapps


def send_whatsapp_message(body: str, to_number: str, cfg):
    tw_cfg = (cfg or {}).get('twilio', {})
    account_sid = tw_cfg.get('account_sid')
    auth_token = tw_cfg.get('auth_token')
    wa_from = tw_cfg.get('whatsapp_from')  # 'whatsapp:+1XXXXXXXXXX'

    if not (account_sid and auth_token and wa_from and to_number):
        return 'skipped', 'Twilio WhatsApp not configured'

    try:
        from twilio.rest import Client  # type: ignore
    except Exception as e:
        return 'error', f"twilio package not installed: {e}"

    try:
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            body=body,
            from_=wa_from,
            to=f"whatsapp:{to_number}" if not str(to_number).startswith('whatsapp:') else to_number,
        )
        return 'sent', f"Message SID: {msg.sid}"
    except Exception as e:
        return 'error', str(e)


if __name__ == "__main__":
    # Load image
    img = cv2.imread(CAR_IMAGE)
    if img is None:
        raise FileNotFoundError(f"Sample image not found at '{CAR_IMAGE}'.")

    display(img, 'Input image')

    # Detect plate
    output_img, plate = detect_plate(img)
    if plate is None:
        display(output_img, 'No plate detected')
        raise SystemExit('No plate detected.')

    display(output_img, 'Detected license plate in the input image')
    display(plate, 'Extracted license plate from the image')

    # Segment characters
    chars = segment_characters(plate)
    if chars.size == 0:
        raise SystemExit('No characters segmented from plate.')

    # Visualize segmented characters (first up to 10)
    n_show = min(10, len(chars))
    for i in range(n_show):
        plt.subplot(1, n_show, i + 1)
        plt.imshow(chars[i], cmap='gray')
        plt.axis('off')
    plt.suptitle('Segmented characters')
    plt.show()

    # Prepare data generators
    has_dataset = os.path.isdir(os.path.join(DATASET_DIR, 'train')) and os.path.isdir(os.path.join(DATASET_DIR, 'val'))
    if not has_dataset:
        raise FileNotFoundError(
            f"Expected dataset directories at '{DATASET_DIR}/train' and '{DATASET_DIR}/val'."
        )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, width_shift_range=0.1, height_shift_range=0.1)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size=(28, 28),
        batch_size=1,
        class_mode='sparse',
    )

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size=(28, 28),
        batch_size=1,
        class_mode='sparse',
    )

    # Build and train model
    model = build_model()
    model.summary()

    class StopTrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if logs.get('val_custom_f1score', 0) > 0.99:
                self.model.stop_training = True

    callbacks = [StopTrainingCallback()]
    batch_size = 1

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        epochs=80,
        verbose=1,
        callbacks=callbacks,
    )

    # Predict plate text
    plate_number = show_results(model, chars)
    print(plate_number)

    # Register detection in DB and send notification
    conn = init_db(DB_PATH)
    detection_id = register_detection(conn, plate_number, CAR_IMAGE)

    cfg = load_config(CONFIG_PATH)
    subject = f"LPR Detection: {plate_number}"
    body = (
        f"A license plate was detected.\n\n"
        f"Plate: {plate_number}\n"
        f"Image: {os.path.abspath(CAR_IMAGE)}\n"
        f"Detection ID: {detection_id}\n"
        f"Timestamp (UTC): {datetime.utcnow().isoformat(timespec='seconds')}Z"
    )

    emails, whatsapps = lookup_recipients(conn, plate_number)

    # Email (owners table first; if none, fall back is to skip)
    if emails:
        status, details = send_email_notification(subject, body, emails, cfg)
        record_notification(conn, detection_id, 'email', status, details)
    else:
        record_notification(conn, detection_id, 'email', 'skipped', 'No registered email for plate')

    # WhatsApp per number
    if whatsapps:
        for wa in whatsapps:
            w_status, w_details = send_whatsapp_message(body, wa, cfg)
            record_notification(conn, detection_id, 'whatsapp', w_status, w_details)
    else:
        record_notification(conn, detection_id, 'whatsapp', 'skipped', 'No registered WhatsApp for plate')

    # Show predicted characters
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(chars[:12]):
        img_disp = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        plt.subplot(3, 4, i + 1)
        plt.imshow(img_disp, cmap='gray')
        plt.title(f'predicted: {plate_number[i] if i < len(plate_number) else "?"}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Overlay prediction on original image
    output_img, _ = detect_plate(img, plate_number)
    display(output_img, 'Detected license plate number in the input image')
