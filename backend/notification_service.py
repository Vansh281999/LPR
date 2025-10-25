import requests
import logging
# Twilio import commented out until installed
# from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class NotificationService:
    def __init__(self, twilio_account_sid=None, twilio_auth_token=None, 
                 telegram_bot_token=None, email_username=None, email_password=None):
        self.logger = logging.getLogger(__name__)
        
        # Twilio for WhatsApp
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_client = None
        # Commented out until Twilio is installed
        # if twilio_account_sid and twilio_auth_token:
        #     self.twilio_client = Client(twilio_account_sid, twilio_auth_token)
            
        # Telegram
        self.telegram_bot_token = telegram_bot_token
        self.telegram_api_url = f"https://api.telegram.org/bot{telegram_bot_token}" if telegram_bot_token else None
        
        # Email
        self.email_username = email_username
        self.email_password = email_password
        
    def send_whatsapp_notification(self, to_number, message):
        """Send WhatsApp notification using Twilio"""
        if not self.twilio_client:
            self.logger.error("Twilio client not configured")
            return False
            
        try:
            # Format the number for WhatsApp
            whatsapp_number = f"whatsapp:{to_number}"
            
            # Send message
            message = self.twilio_client.messages.create(
                from_='whatsapp:+14155238886',  # Twilio sandbox number
                body=message,
                to=whatsapp_number
            )
            
            self.logger.info(f"WhatsApp notification sent to {to_number}, SID: {message.sid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send WhatsApp notification: {str(e)}")
            return False
            
    def send_telegram_notification(self, chat_id, message):
        """Send Telegram notification"""
        if not self.telegram_api_url:
            self.logger.error("Telegram bot not configured")
            return False
            
        try:
            # Send message
            send_message_url = f"{self.telegram_api_url}/sendMessage"
            response = requests.post(
                send_message_url,
                json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
            )
            
            if response.status_code == 200:
                self.logger.info(f"Telegram notification sent to chat ID {chat_id}")
                return True
            else:
                self.logger.error(f"Failed to send Telegram notification: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Telegram notification: {str(e)}")
            return False
            
    def send_email_notification(self, to_email, subject, message):
        """Send email notification"""
        if not self.email_username or not self.email_password:
            self.logger.error("Email credentials not configured")
            return False
            
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.email_username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Attach message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_username, self.email_password)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent to {to_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
            return False
            
    def notify_vehicle_owner(self, owner_info, message_type, plate_number=None, accuracy=None):
        """Notify vehicle owner through available channels"""
        success = False
        
        # Prepare message based on type
        if message_type == "authorized_entry":
            message = f"Your vehicle with license plate {plate_number} has been authorized for entry."
            if accuracy:
                message += f" (Recognition accuracy: {accuracy:.1f}%)"
        elif message_type == "unauthorized_attempt":
            message = f"Alert: An unauthorized vehicle with license plate {plate_number} attempted entry."
        else:
            message = f"Notification regarding your vehicle with license plate {plate_number}."
            
        # Try WhatsApp
        if owner_info.get('whatsapp_number'):
            whatsapp_success = self.send_whatsapp_notification(
                owner_info['whatsapp_number'], 
                message
            )
            success = success or whatsapp_success
            
        # Try Telegram
        if owner_info.get('telegram_chat_id'):
            telegram_success = self.send_telegram_notification(
                owner_info['telegram_chat_id'],
                message
            )
            success = success or telegram_success
            
        # Try Email as fallback
        if owner_info.get('email') and not success:
            email_success = self.send_email_notification(
                owner_info['email'],
                f"Vehicle Notification - {message_type.replace('_', ' ').title()}",
                message
            )
            success = success or email_success
            
        return success