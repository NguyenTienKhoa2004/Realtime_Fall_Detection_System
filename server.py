from flask import Flask, request, jsonify, render_template, send_from_directory, abort
import os
import json
from datetime import datetime
import smtplib
import ssl
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import os
from dotenv import load_dotenv

load_dotenv()

# ==================================================
# --- CẤU HÌNH GMAIL ---
# ==================================================
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER')

ALERT_SUBJECT = "ALERT_FALL_DETECTED" 

# ==================================================
# --- CẤU HÌNH HỆ THỐNG ---
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'evidence_image')
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'), 
            static_folder=os.path.join(BASE_DIR, 'static'))

ALERT_HISTORY = []
MAX_HISTORY = 200

# ==================================================
# --- HÀM GỬI EMAIL CHẠY NGẦM ---
# ==================================================
def send_email_thread(label, prob, image_path, time_str, meta):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = ALERT_SUBJECT  
        
        body = f"""
        ⚠️ CẢNH BÁO TÉ NGÃ KHẨN CẤP
        ---------------------------
        - Trạng thái: {label} (Xác nhận ngã và bất động)
        - Thời gian: {time_str}
        - Độ tin cậy: {prob * 100:.2f}%
        - Camera ID: {meta.get('camera_id', 'Unknown')}
        - Chỉ số chuyển động: {meta.get('motion', 'N/A')}
        
        (Email này dùng để kích hoạt Phím tắt gửi SMS trên iPhone)
        """
        msg.attach(MIMEText(body, 'plain'))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= evidence_{datetime.now().strftime('%H%M%S')}.jpg",
            )
            msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            
        print(f"[SUCCESS] Đã gửi tín hiệu báo động đến {EMAIL_RECEIVER}")

    except Exception as e:
        print(f"[EMAIL ERROR] Lỗi gửi mail: {str(e)}")

# ==================================================
# --- CÁC ĐIỂM CUỐI (ENDPOINTS) ---
# ==================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify(ALERT_HISTORY)

@app.route('/evidence/<path:filename>')
def serve_evidence(filename):
    safe_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(safe_path):
        abort(404)
    return send_from_directory(SAVE_DIR, filename)

@app.route('/alert', methods=['POST'])
def receive_alert():
    try:
        label = request.form.get('label', 'Unknown')
        prob_raw = request.form.get('prob', '0.0')
        prob = float(prob_raw)

        meta = {}
        meta_field = request.form.get('meta')
        if meta_field:
            meta = json.loads(meta_field)

        filename_saved = 'no_image.jpg'
        full_image_path = None
        if 'image' in request.files:
            f = request.files['image']
            if f and f.filename:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename_saved = f'alert_{stamp}.jpg'
                full_image_path = os.path.join(SAVE_DIR, filename_saved)
                f.save(full_image_path)

        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        record = {
            'time': time_str,
            'label': label,
            'prob': prob,
            'image': filename_saved,
            'meta': meta,
        }
        ALERT_HISTORY.append(record)
        if len(ALERT_HISTORY) > MAX_HISTORY:
            ALERT_HISTORY.pop(0)

        print(f"[ALERT RECEIVED] State: {label} | Prob: {prob}")

        if str(label).lower() in ["fall", "1", "confirmed", "inactive", "fall_confirmed"]:
            print("--> Đang khởi chạy luồng gửi mail báo động...")
            t = threading.Thread(
                target=send_email_thread, 
                args=(label, prob, full_image_path, time_str, meta)
            )
            t.start()

        return jsonify({'status': 'ok'}), 201

    except Exception as e:
        print('[SERVER ERROR]:', e)
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    ALERT_HISTORY.clear()
    return jsonify({'status': 'cleared'}), 200

if __name__ == '__main__':
    print(f"Server đang chạy tại cổng 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)