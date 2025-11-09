#!/usr/bin/env python3
"""
Interactive setup script for LPR notification system.
Helps configure SMTP and Twilio credentials.
"""

import json
import os
import getpass
from pathlib import Path

CONFIG_PATH = 'config.json'

def load_existing_config():
    """Load existing configuration if it exists."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    
    return {
        "smtp": {
            "host": "",
            "port": 587,
            "user": "",
            "password": "",
            "sender": ""
        },
        "twilio": {
            "account_sid": "",
            "auth_token": "",
            "whatsapp_from": ""
        }
    }

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def setup_smtp(config):
    """Configure SMTP settings."""
    print("\n=== SMTP Email Configuration ===")
    print("For Gmail: smtp.gmail.com, port 587")
    print("For Outlook: smtp-mail.outlook.com, port 587")
    
    current = config.get('smtp', {})
    
    host = input(f"SMTP Host [{current.get('host', '')}]: ").strip()
    if host:
        config['smtp']['host'] = host
    
    port = input(f"SMTP Port [{current.get('port', 587)}]: ").strip()
    if port:
        config['smtp']['port'] = int(port)
    
    user = input(f"SMTP Username [{current.get('user', '')}]: ").strip()
    if user:
        config['smtp']['user'] = user
    
    password = getpass.getpass("SMTP Password (hidden): ").strip()
    if password:
        config['smtp']['password'] = password
    
    sender = input(f"From Email [{current.get('sender', '')}]: ").strip()
    if sender:
        config['smtp']['sender'] = sender

def setup_twilio(config):
    """Configure Twilio WhatsApp settings."""
    print("\n=== Twilio WhatsApp Configuration ===")
    print("Get these from: https://console.twilio.com/")
    print("WhatsApp sandbox number format: whatsapp:+14155238886")
    
    current = config.get('twilio', {})
    
    account_sid = input(f"Twilio Account SID [{current.get('account_sid', '')}]: ").strip()
    if account_sid:
        config['twilio']['account_sid'] = account_sid
    
    auth_token = getpass.getpass("Twilio Auth Token (hidden): ").strip()
    if auth_token:
        config['twilio']['auth_token'] = auth_token
    
    whatsapp_from = input(f"WhatsApp From Number [{current.get('whatsapp_from', '')}]: ").strip()
    if whatsapp_from:
        config['twilio']['whatsapp_from'] = whatsapp_from

def register_sample_owner(config):
    """Register a sample owner for testing."""
    print("\n=== Register Sample Owner ===")
    
    try:
        from owner_admin import upsert_owner
        
        plate = input("Enter a test license plate (e.g., DL8CAF5030): ").strip().upper()
        if not plate:
            print("Skipping owner registration.")
            return
        
        email = input("Enter email address: ").strip()
        whatsapp = input("Enter WhatsApp number (+91XXXXXXXXXX): ").strip()
        
        if email or whatsapp:
            upsert_owner(plate, email, whatsapp)
            print(f"✅ Registered owner for plate: {plate}")
        else:
            print("No contact details provided, skipping registration.")
    except ImportError:
        print("❌ Could not import owner_admin module.")
    except Exception as e:
        print(f"❌ Error registering owner: {e}")

def test_notifications(config):
    """Test the notification system."""
    print("\n=== Test Notifications ===")
    
    # Test email
    if config.get('smtp', {}).get('host'):
        print("Testing email...")
        try:
            from lpr import send_email_notification
            status, details = send_email_notification(
                "LPR Test Email",
                "This is a test notification from your LPR system.",
                [config['smtp'].get('sender')],  # Send to self for testing
                config
            )
            print(f"Email test: {status} - {details}")
        except Exception as e:
            print(f"❌ Email test failed: {e}")
    
    # Test WhatsApp
    if config.get('twilio', {}).get('account_sid'):
        test_number = input("Enter WhatsApp number to test (+91XXXXXXXXXX): ").strip()
        if test_number:
            print("Testing WhatsApp...")
            try:
                from lpr import send_whatsapp_message
                status, details = send_whatsapp_message(
                    "This is a test notification from your LPR system.",
                    test_number,
                    config
                )
                print(f"WhatsApp test: {status} - {details}")
            except Exception as e:
                print(f"❌ WhatsApp test failed: {e}")

def main():
    """Main setup function."""
    print("🚗 LPR Notification System Setup")
    print("=" * 40)
    
    config = load_existing_config()
    
    while True:
        print("\nOptions:")
        print("1. Configure SMTP (Email)")
        print("2. Configure Twilio (WhatsApp)")
        print("3. Register/Update Owner")
        print("4. Test Notifications")
        print("5. Save & Exit")
        print("6. Exit without saving")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == '1':
            setup_smtp(config)
        elif choice == '2':
            setup_twilio(config)
        elif choice == '3':
            register_sample_owner(config)
        elif choice == '4':
            test_notifications(config)
        elif choice == '5':
            save_config(config)
            print(f"✅ Configuration saved to {CONFIG_PATH}")
            break
        elif choice == '6':
            print("Exiting without saving changes.")
            break
        else:
            print("Invalid option. Please choose 1-6.")

if __name__ == "__main__":
    main()