from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import mysql.connector
import pickle
import smtplib
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import os
import logging
from scipy import stats
from flask_mail import Mail, Message
import secrets
import string
from fpdf import FPDF
import io

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MySQL configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'rootpassword',
    'database': 'student_grade_system'
}

# Model file paths
MODEL_DIR = '/workspace/models'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, 'target_scaler.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_student_performance_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_student_performance_model.pkl')
META_MODEL_PATH = os.path.join(MODEL_DIR, 'meta_model.pkl')

# Database initialization
def init_db():
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS student_grade_system")
        cursor.close()
        conn.close()

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS loginmaster (
            prn VARCHAR(50) PRIMARY KEY,
            password VARCHAR(255) NOT NULL,
            user_type VARCHAR(20) NOT NULL
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS studentdetails (
            prn VARCHAR(50) PRIMARY KEY,
            roll_number VARCHAR(50),
            name VARCHAR(100) NOT NULL,
            panel VARCHAR(50),
            email VARCHAR(100) NOT NULL
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS marks_master (
            prn VARCHAR(50) PRIMARY KEY,
            cca1 FLOAT,
            cca2 FLOAT,
            cca3 FLOAT,
            lca1 FLOAT,
            lca2 FLOAT,
            lca3 FLOAT,
            co1 FLOAT,
            co2 FLOAT,
            co3 FLOAT,
            co4 FLOAT,
            FOREIGN KEY (prn) REFERENCES studentdetails(prn)
        )''')
        
        # Add new table for prediction history to track fluctuations
        cursor.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            prn VARCHAR(50),
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_score FLOAT,
            prediction_lower_bound FLOAT,
            prediction_upper_bound FLOAT,
            FOREIGN KEY (prn) REFERENCES studentdetails(prn)
        )''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database initialized successfully.")
    except mysql.connector.Error as err:
        logger.error(f"Error initializing database: {err}")
        raise

init_db()

# Load the ML models and scalers with detailed logging
scaler = None
target_scaler = None
rf_model = None
xgb_model = None
meta_model = None

def load_model(file_path, loader_func, model_name):
    if not os.path.exists(file_path):
        logger.error(f"{model_name} file not found at {file_path}")
        return None
    try:
        model = loader_func(file_path)
        logger.info(f"{model_name} loaded successfully from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name} from {file_path}: {e}")
        return None

try:
    scaler = load_model(SCALER_PATH, lambda x: pickle.load(open(x, 'rb')), "Scaler")
    target_scaler = load_model(TARGET_SCALER_PATH, lambda x: pickle.load(open(x, 'rb')), "Target Scaler")
    rf_model = load_model(RF_MODEL_PATH, lambda x: pickle.load(open(x, 'rb')), "Random Forest Model")
    xgb_model = load_model(XGB_MODEL_PATH, lambda x: pickle.load(open(x, 'rb')), "XGBoost Model")
    meta_model = load_model(META_MODEL_PATH, lambda x: pickle.load(open(x, 'rb')), "Meta Model")
    
    if all([scaler, target_scaler, rf_model, xgb_model, meta_model]):
        logger.info("All ML models and scalers loaded successfully.")
    else:
        logger.warning("One or more models/scalers failed to load. Prediction will be skipped.")

except Exception as e:
    logger.error(f"Unexpected error during model loading: {e}")

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    prn = data['prn']
    password = data['password']
    user_type = data['userType']

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM loginmaster WHERE prn = %s AND password = %s AND user_type = %s', (prn, password, user_type))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            return jsonify({'success': True, 'user': {'prn': prn, 'userType': user_type}})
        return jsonify({'success': False})
    except mysql.connector.Error as err:
        logger.error(f"Error during login: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500

@app.route('/api/add-user', methods=['POST'])
def add_user():
    data = request.get_json()
    prn = data['prn']
    password = data['password']
    user_type = data['userType']
    name = data['name']
    email = data['email']

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO loginmaster (prn, password, user_type) VALUES (%s, %s, %s)', (prn, password, user_type))
        
        if user_type == 'student':
            roll_number = data['rollNumber']
            panel = data['panel']
            cursor.execute('INSERT INTO studentdetails (prn, roll_number, name, panel, email) VALUES (%s, %s, %s, %s, %s)', 
                           (prn, roll_number, name, panel, email))
        else:
            cursor.execute('INSERT INTO studentdetails (prn, roll_number, name, panel, email) VALUES (%s, %s, %s, %s, %s)', 
                           (prn, '', name, '', email))
        
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except mysql.connector.Error as err:
        logger.error(f"Error adding user: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('SELECT prn, name, email FROM studentdetails WHERE prn IN (SELECT prn FROM loginmaster WHERE user_type = "student")')
        students = [{'prn': row[0], 'name': row[1], 'email': row[2]} for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return jsonify({'students': students})
    except mysql.connector.Error as err:
        logger.error(f"Error fetching students: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500

@app.route('/api/add-marks', methods=['POST'])
def add_marks():
    data = request.get_json()
    prn = data['prn']
    
    # Extract marks (convert to float, handling empty strings)
    def get_float_value(field):
        value = data.get(field, '')
        return float(value) if value != '' else None
    
    cca1 = get_float_value('cca1')
    cca2 = get_float_value('cca2')
    cca3 = get_float_value('cca3')
    lca1 = get_float_value('lca1')
    lca2 = get_float_value('lca2')
    lca3 = get_float_value('lca3')
    co1 = get_float_value('co1')
    co2 = get_float_value('co2')
    co3 = get_float_value('co3')
    co4 = get_float_value('co4')
    
    # Process absent reasons and create remarks
    absent_reasons = data.get('absentReasons', {})
    remarks = []
    
    for field, reason in absent_reasons.items():
        if reason:  # If there's a reason for this field
            remarks.append(f"{field.upper()}: {reason}")
    
    # Combine all remarks into a single string
    remark = "; ".join(remarks) if remarks else None

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Updated query to include remark field
        cursor.execute('''INSERT INTO marks_master 
                          (prn, cca1, cca2, cca3, lca1, lca2, lca3, co1, co2, co3, co4, remark) 
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                          ON DUPLICATE KEY UPDATE 
                          cca1=VALUES(cca1), cca2=VALUES(cca2), cca3=VALUES(cca3), 
                          lca1=VALUES(lca1), lca2=VALUES(lca2), lca3=VALUES(lca3), 
                          co1=VALUES(co1), co2=VALUES(co2), co3=VALUES(co3), co4=VALUES(co4),
                          remark=VALUES(remark)''', 
                       (prn, cca1, cca2, cca3, lca1, lca2, lca3, co1, co2, co3, co4, remark))
        
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
        
    except mysql.connector.Error as err:
        logger.error(f"Error adding marks: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500
    except ValueError as ve:
        logger.error(f"Invalid mark value: {ve}")
        return jsonify({'success': False, 'error': 'Invalid mark value'}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'success': False, 'error': 'Server error'}), 500

def generate_prediction_with_uncertainty(base_prediction, confidence_level=0.90):
    """
    Generate a prediction with confidence intervals to reflect uncertainty
    """
    # Standard deviation is proportional to distance from extremes (more uncertain in the middle)
    # This creates more realistic fluctuations based on the prediction value
    if base_prediction <= 50:
        std_dev = base_prediction * 0.15  # Higher uncertainty for lower performers
    else:
        std_dev = (100 - base_prediction) * 0.12  # Lower uncertainty for high performers
    
    # Create a truncated normal distribution between 0-100
    a, b = (0 - base_prediction) / std_dev, (100 - base_prediction) / std_dev
    lower_bound = max(0, base_prediction - stats.truncnorm.ppf(1 - (1 - confidence_level)/2, a, b) * std_dev)
    upper_bound = min(100, base_prediction + stats.truncnorm.ppf(1 - (1 - confidence_level)/2, a, b) * std_dev)
    
    # Add small random fluctuation to make predictions more dynamic
    fluctuation = np.random.normal(0, std_dev * 0.2)
    adjusted_prediction = min(100, max(0, base_prediction + fluctuation))
    
    return adjusted_prediction, lower_bound, upper_bound

@app.route('/api/student-graph/<prn>', methods=['GET'])
def student_graph(prn):
    logger.info(f"Processing student graph for PRN: {prn}")
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get all student data including remarks in one query
        cursor.execute('''
            SELECT cca1, cca2, cca3, lca1, lca2, lca3, co1, co2, co3, co4, remark
            FROM marks_master 
            WHERE prn = %s
        ''', (prn,))
        student_data = cursor.fetchone()
        
        # Get prediction data
        cursor.execute('''
            SELECT predicted_score, prediction_lower_bound, prediction_upper_bound, prediction_date 
            FROM prediction_history 
            WHERE prn = %s 
            ORDER BY prediction_date DESC 
            LIMIT 1
        ''', (prn,))
        prev_prediction = cursor.fetchone()
        
        cursor.close()
        conn.close()

        if not student_data or all(m == 0 or m is None for m in [student_data.get('cca1'), student_data.get('cca2'), student_data.get('cca3'), 
                                                               student_data.get('lca1'), student_data.get('lca2'), student_data.get('lca3')]):
            logger.warning(f"No valid marks found for PRN: {prn}")
            return jsonify({
                'success': True,
                'status': 'No Data',
                'overallScore': 0.0,
                'predictedScore': 0.0,
                'lowerBound': 0.0,
                'upperBound': 0.0,
                'threshold': 50,
                'remark': student_data.get('remark', 'No remarks available') if student_data else 'No data found',
                'hasRemark': bool(student_data and student_data.get('remark'))
            })

        # Extract marks and calculate scores (your existing logic)
        marks = [float(student_data[key]) if student_data[key] is not None else 0.0 
                for key in ['cca1', 'cca2', 'cca3', 'lca1', 'lca2', 'lca3']]
        overall_score = sum(marks[:6])
        logger.debug(f"Raw overall score for PRN {prn}: {overall_score}")

        avg_cca = sum(marks[:3]) / 3
        avg_lca = sum(marks[3:6]) / 3
        feature_names = ['cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
                         'lca_1_practical_performance', 'lca_2_active_learning_project',
                         'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca']
        marks_with_derived = list(marks[:6]) + [avg_cca, avg_lca]

        max_possible_score = 50
        overall_score_scaled = min((overall_score / max_possible_score) * 100, 100)

        if scaler and target_scaler and rf_model and xgb_model and meta_model:
            try:
                marks_df = pd.DataFrame([marks_with_derived], columns=feature_names)
                scaled_marks = scaler.transform(marks_df)
                
                rf_pred = rf_model.predict(scaled_marks)[0]
                xgb_pred = xgb_model.predict(scaled_marks)[0]
                
                stacked_features = np.array([[rf_pred, xgb_pred, overall_score_scaled]])
                logger.debug(f"Meta model input shape: {stacked_features.shape}")
                
                stacked_pred = meta_model.predict(stacked_features)[0]
                base_prediction = float(target_scaler.inverse_transform([[stacked_pred]])[0][0])
                
                if prev_prediction:
                    prev_score = float(prev_prediction[0])
                    consistency_factor = 0.7
                    base_prediction = (consistency_factor * base_prediction) + ((1 - consistency_factor) * prev_score)
                
                adjustment_factor = 0.7
                base_prediction = (adjustment_factor * base_prediction) + ((1 - adjustment_factor) * overall_score_scaled)
                base_prediction = min(base_prediction, 100)
                
            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                base_prediction = overall_score_scaled
            
            predicted_score, lower_bound, upper_bound = generate_prediction_with_uncertainty(base_prediction)
            logger.info(f"Predicted Score for PRN {prn}: {predicted_score} [{lower_bound}-{upper_bound}]")
            
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO prediction_history (prn, predicted_score, prediction_lower_bound, prediction_upper_bound) '
                           'VALUES (%s, %s, %s, %s)', (prn, predicted_score, lower_bound, upper_bound))
            conn.commit()
            cursor.close()
            conn.close()
        else:
            predicted_score = overall_score_scaled
            lower_bound = max(0, predicted_score - 10)
            upper_bound = min(100, predicted_score + 10)
            logger.warning(f"Prediction skipped for PRN {prn} due to model loading error.")

        # Determine performance status
        if predicted_score >= 80:
            status = "Excellent"
        elif predicted_score >= 60:
            status = "Good"
        elif predicted_score >= 50:
            status = "Average"
        else:
            status = "Needs Improvement"

        return jsonify({
            'success': True,
            'status': status,
            'overallScore': float(overall_score_scaled),
            'predictedScore': float(predicted_score),
            'lowerBound': float(lower_bound),
            'upperBound': float(upper_bound),
            'threshold': 50,
            'remark': student_data.get('remark', 'No remarks available'),
            'hasRemark': bool(student_data.get('remark'))
        })

    except Exception as e:
        logger.error(f"Error in student graph prediction for PRN {prn}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'Error',
            'overallScore': float(overall_score_scaled if 'overall_score_scaled' in locals() else 0),
            'predictedScore': 0.0,
            'lowerBound': 0.0,
            'upperBound': 0.0,
            'threshold': 50,
            'remark': 'Error fetching remarks',
            'hasRemark': False
        }), 500

@app.route('/api/send-feedback', methods=['POST'])
def send_feedback():
    data = request.get_json()
    prn = data['prn']
    email = data['email']
    feedback = data['feedback']

    try:
        msg = MIMEText(feedback)
        msg['Subject'] = 'Feedback from Teacher'
        msg['From'] = 'aayush.shah@mitwpu.edu.in'
        msg['To'] = email

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('aayush.shah@mitwpu.edu.in', 'flsu hpwx lrjk nnwd')
            server.sendmail('aayush.shah@mitwpu.edu.in', email, msg.as_string())

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error sending feedback: {e}")
        return jsonify({'success': False, 'error': 'Email sending failed'}), 500

@app.route('/api/submit-discrepancy', methods=['POST'])
def submit_discrepancy():
    data = request.get_json()
    prn = data['prn']
    discrepancy = data['discrepancy']
    to_email = data['toEmail']

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('SELECT email FROM studentdetails WHERE prn = %s', (prn,))
        student_email = cursor.fetchone()
        if not student_email:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Student email not found'}), 404
        from_email = student_email[0]
        cursor.close()
        conn.close()

        msg = MIMEText(f'Discrepancy from PRN {prn}:\n\n{discrepancy}')
        msg['Subject'] = 'Discrepancy Report'
        msg['From'] = from_email
        msg['To'] = to_email

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('aayush.shah@mitwpu.edu.in', 'flsu hpwx lrjk nnwd')
            server.sendmail(from_email, to_email, msg.as_string())

        return jsonify({'success': True})
    except mysql.connector.Error as err:
        logger.error(f"Error fetching student email: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500
    except Exception as e:
        logger.error(f"Error submitting discrepancy: {e}")
        return jsonify({'success': False, 'error': 'Email sending failed'}), 500

# Configure Flask-Mail (add to your existing Flask config)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'aayush.shah@mitwpu.edu.in'
app.config['MAIL_PASSWORD'] = 'flsu hpwx lrjk nnwd'
mail = Mail(app)

# Add this route to your existing Flask app
@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data['email']
    user_type = data.get('userType', 'student')
    
    conn = None
    cursor = None
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Check if email exists based on user type
        if user_type == 'student':
            query = 'SELECT prn, email FROM studentdetails WHERE email = %s'
            cursor.execute(query, (email,))
        else:
            query = '''
                SELECT l.prn, s.email 
                FROM loginmaster l
                JOIN studentdetails s ON l.prn = s.prn
                WHERE s.email = %s AND l.user_type = %s
            '''
            cursor.execute(query, (email, user_type))
        
        user = cursor.fetchone()
        
        # Ensure we consume all results
        while cursor.nextset():
            pass
        
        if not user:
            return jsonify({'success': False, 'message': 'Email not found for this user type'}), 404
        
        # Generate reset token
        reset_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        
        # Store token in database
        update_query = '''
            UPDATE loginmaster 
            SET reset_token = %s, token_expiry = DATE_ADD(NOW(), INTERVAL 1 HOUR)
            WHERE prn = %s
        '''
        cursor.execute(update_query, (reset_token, user['prn']))
        conn.commit()
        
        # Send email
        reset_link = f"http://localhost:3000/reset-password?token={reset_token}"
        msg = Message('Password Reset Request',
                     sender='noreply@yourdomain.com',
                     recipients=[email])
        msg.body = f'''
            To reset your password, click the following link:
            {reset_link}
            
            This link will expire in 1 hour.
            
            If you didn't request this, please ignore this email.
        '''
        mail.send(msg)
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500
    finally:
        if cursor:
            try:
                # Ensure all results are consumed
                while cursor.nextset():
                    pass
                cursor.close()
            except:
                pass
        if conn and conn.is_connected():
            conn.close()

@app.route('/api/generate-student-report/<prn>', methods=['GET'])
def generate_student_report(prn):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get student details
        cursor.execute('''
            SELECT s.prn, s.name, s.email, s.panel, m.* 
            FROM studentdetails s
            LEFT JOIN marks_master m ON s.prn = m.prn
            WHERE s.prn = %s
        ''', (prn,))
        student = cursor.fetchone()
        
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'}), 404
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Title
        pdf.cell(0, 10, f'Student Performance Report: {student["name"]}', 0, 1, 'C')
        pdf.ln(10)
        
        # Student Info
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'PRN: {student["prn"]}', 0, 1)
        pdf.cell(0, 10, f'Email: {student["email"]}', 0, 1)
        pdf.cell(0, 10, f'Panel: {student["panel"]}', 0, 1)
        pdf.ln(10)
        
        # Marks Table
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Assessment Marks', 0, 1)
        pdf.set_font('Arial', 'B', 12)
        
        # Table Header
        pdf.cell(90, 10, 'Assessment', 1, 0, 'C')
        pdf.cell(90, 10, 'Marks', 1, 1, 'C')
        pdf.set_font('Arial', '', 12)
        
        # Table Rows
        assessments = [
            ('CCA-1 (10 marks)', student.get('cca1', 'N/A')),
            ('CCA-2 (5 marks)', student.get('cca2', 'N/A')),
            ('CCA-3 (Mid term, 15 marks)', student.get('cca3', 'N/A')),
            ('LCA-1 (Practical Performance)', student.get('lca1', 'N/A')),
            ('LCA-2 (Active Learning/Project)', student.get('lca2', 'N/A')),
            ('LCA-3 (End term practical/oral)', student.get('lca3', 'N/A'))
        ]
        
        for name, mark in assessments:
            pdf.cell(90, 10, name, 1, 0)
            pdf.cell(90, 10, str(mark), 1, 1, 'C')
        
        # Generate PDF to memory
        pdf_output = io.BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Generate PDF as a byte string
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        
        return send_file(
            pdf_output,
            as_attachment=True,
            download_name=f'Student_Report_{prn}.pdf',
            mimetype='application/pdf'
        ), 200
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'success': False, 'message': 'Error generating report'}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)
