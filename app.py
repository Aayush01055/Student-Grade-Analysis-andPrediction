from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import pickle
import smtplib
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MySQL configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'student_grade_system'
}

# Model file paths
MODEL_DIR = 'C:/Users/samik/Downloads/ML/ML/models'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, 'target_scaler.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_student_performance_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_student_performance_model.pkl')
NN_MODEL_PATH = os.path.join(MODEL_DIR, 'nn_student_performance_model.h5')
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
nn_model = None
meta_model = None

def load_model(file_path, loader_func, model_name):
    if not os.path.exists(file_path):
        logger.error(f"{model_name} file not found at {file_path}")
        return None
    try:
        if model_name == "Neural Network Model":
            # Register 'mse' as a custom object
            @tf.keras.saving.register_keras_serializable()
            def mse(y_true, y_pred):
                return tf.keras.losses.mean_squared_error(y_true, y_pred)
            custom_objects = {'mse': mse}
            model = loader_func(file_path, custom_objects=custom_objects)
        else:
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
    nn_model = load_model(NN_MODEL_PATH, tf.keras.models.load_model, "Neural Network Model")
    meta_model = load_model(META_MODEL_PATH, lambda x: pickle.load(open(x, 'rb')), "Meta Model")
    
    if all([scaler, target_scaler, rf_model, xgb_model, nn_model, meta_model]):
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
    cca1 = float(data['cca1'])
    cca2 = float(data['cca2'])
    cca3 = float(data['cca3'])
    lca1 = float(data['lca1'])
    lca2 = float(data['lca2'])
    lca3 = float(data['lca3'])
    co1 = float(data['co1'])
    co2 = float(data['co2'])
    co3 = float(data['co3'])
    co4 = float(data['co4'])

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO marks_master 
                          (prn, cca1, cca2, cca3, lca1, lca2, lca3, co1, co2, co3, co4) 
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                          ON DUPLICATE KEY UPDATE 
                          cca1=%s, cca2=%s, cca3=%s, lca1=%s, lca2=%s, lca3=%s, co1=%s, co2=%s, co3=%s, co4=%s''', 
                       (prn, cca1, cca2, cca3, lca1, lca2, lca3, co1, co2, co3, co4,
                        cca1, cca2, cca3, lca1, lca2, lca3, co1, co2, co3, co4))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except mysql.connector.Error as err:
        logger.error(f"Error adding marks: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500

@app.route('/api/student-graph/<prn>', methods=['GET'])
def student_graph(prn):
    logger.info(f"Processing student graph for PRN: {prn}")
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute('SELECT cca1, cca2, cca3, '
                      'lca1, lca2, lca3, '
                      'co1, co2, co3, co4 '
                      'FROM marks_master WHERE prn = %s', (prn,))
        marks = cursor.fetchone()
        cursor.close()
        conn.close()

        if not marks:
            logger.warning(f"No marks found for PRN: {prn}")
            return jsonify({'overallScore': 0, 'predictedScore': 0, 'threshold': 50})

        # Ensure marks are floats and handle None values
        marks = [float(m) if m is not None else 0.0 for m in marks]
        
        # Calculate overall score (sum of CCA and LCA components)
        overall_score = sum(marks[:6])  # Sum of cca1, cca2, cca3, lca1, lca2, lca3
        logger.debug(f"Raw overall score for PRN {prn}: {overall_score}")

        # Compute derived features
        avg_cca = sum(marks[:3]) / 3
        avg_lca = sum(marks[3:6]) / 3

        # Prepare input features
        feature_names = [
            'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
            'lca_1_practical_performance', 'lca_2_active_learning_project',
            'lca_3_end_term_practical_oral',
            'avg_cca', 'avg_lca'
        ]
        marks_with_derived = list(marks[:6]) + [avg_cca, avg_lca]

        # Scale overall_score to 0-100
        max_possible_score = 50
        overall_score_scaled = min((overall_score / max_possible_score) * 100, 100)

        # Prediction logic
        if (scaler is not None and target_scaler is not None and 
            rf_model is not None and xgb_model is not None and 
            nn_model is not None and meta_model is not None):
            marks_df = pd.DataFrame([marks_with_derived], columns=feature_names)
            scaled_marks = scaler.transform(marks_df)
            
            rf_pred = rf_model.predict(scaled_marks)[0]
            xgb_pred = xgb_model.predict(scaled_marks)[0]
            nn_pred = nn_model.predict(scaled_marks, verbose=0)[0][0]
            
            stacked_X = np.array([[rf_pred, xgb_pred, nn_pred]])
            stacked_pred = meta_model.predict(stacked_X)[0]
            
            predicted_score = target_scaler.inverse_transform([[stacked_pred]])[0][0]
            
            logger.debug(f"PRN: {prn}, RF: {rf_pred}, XGB: {xgb_pred}, NN: {nn_pred}, "
                         f"Stacked: {stacked_pred}, Scaled Prediction: {predicted_score}")

            adjustment_factor = 0.7
            predicted_score = (adjustment_factor * predicted_score) + ((1 - adjustment_factor) * overall_score_scaled)
            predicted_score = min(predicted_score, 100)
            logger.info(f"Adjusted Predicted Score for PRN {prn}: {predicted_score}")
        else:
            predicted_score = overall_score_scaled
            logger.warning(f"Prediction skipped for PRN {prn} due to model loading error.")

        return jsonify({
            'overallScore': float(overall_score_scaled),
            'predictedScore': float(predicted_score),
            'threshold': 50
        })
    except mysql.connector.Error as err:
        logger.error(f"Database error fetching student graph data for PRN {prn}: {err}")
        return jsonify({'success': False, 'error': 'Database error'}), 500
    except Exception as e:
        logger.error(f"Error in student graph prediction for PRN {prn}: {e}")
        return jsonify({'success': False, 'error': 'Prediction error'}), 500

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

if __name__ == '__main__':
    app.run(debug=True)