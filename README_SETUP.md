# LPR Notification System Setup

Your License Plate Recognition system can now send notifications via **Email** and **WhatsApp** when it detects a license plate.

## Quick Setup

### 1. Configure Email (Gmail)

Edit `config.json` file and replace:
- `YOUR_GMAIL_APP_PASSWORD_HERE` with your Gmail App Password

**To get Gmail App Password:**
1. Go to https://myaccount.google.com/
2. Click "Security" → "2-Step Verification" (enable if not already)
3. Click "App passwords" 
4. Select "Mail" and generate password
5. Copy the 16-character password to config.json

### 2. Configure WhatsApp (Optional)

If you want WhatsApp notifications:
1. Sign up at https://console.twilio.com/
2. Get Account SID and Auth Token from dashboard
3. Set up WhatsApp Sandbox
4. Update `config.json` with your Twilio credentials

### 3. Register Vehicle Owners

Register a license plate with owner contact info:

```bash
python owner_admin.py DL8CAF5030 --email owner@example.com --whatsapp +919876543210
```

### 4. Test the System

```bash
python lpr.py
```

The system will:
- Detect license plates in images
- Look up registered owners
- Send email/WhatsApp notifications automatically

## Files Created:

- `config.json` - Your credentials (keep private!)
- `society_vehicles.db` - Database of registered vehicles
- `setup_notifications.py` - Interactive setup tool (optional)

## No More Environment Variables!

Your credentials are now saved in `config.json` - you don't need to set environment variables each time you run the app.