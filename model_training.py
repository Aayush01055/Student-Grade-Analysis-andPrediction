from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import pickle
import numpy as np
import tensorflow as tf
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'student_db'

mysql = MySQL(app)

# Model directory
MODEL_DIR = Path('./models')

# Load trained models
try:
    with open(MODEL_DIR / 'rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open(MODEL_DIR / 'svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)

    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    nn_model = tf.keras.models.load_model(MODEL_DIR / 'nn_model.h5')
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Expected features for prediction
EXPECTED_FEATURES = [
    'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
    'lca_1_practical_performance', 'lca_2_active_learning_project',
    'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca'
]

def validate_input(data, required_fields):
    """Validate input data contains all required fields"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    return True

def prepare_features(data):
    """Prepare and scale features for model prediction"""
    try:
        features = [data[f] for f in EXPECTED_FEATURES]
        return scaler.transform([features])
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise ValueError("Invalid feature data provided")

def db_operation(func):
    """Decorator for database operations with error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            mysql.connection.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            return jsonify({'error': 'Database operation failed'}), 500
    return wrapper

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        validate_input(data, ['prn', 'password'])
        
        prn = data['prn']
        password = data['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT password FROM loginmaster WHERE prn=%s", (prn,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[0], password):
            return jsonify({'message': 'Login successful'})
        return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/add-user', methods=['POST'])
@db_operation
def add_user():
    data = request.json
    validate_input(data, ['prn', 'password'])
    
    prn = data['prn']
    password = generate_password_hash(data['password'])

    cur = mysql.connection.cursor()
    # Check if user already exists
    cur.execute("SELECT prn FROM loginmaster WHERE prn=%s", (prn,))
    if cur.fetchone():
        cur.close()
        return jsonify({'error': 'User already exists'}), 400
    
    cur.execute("INSERT INTO loginmaster (prn, password) VALUES (%s, %s)", (prn, password))
    mysql.connection.commit()
    cur.close()

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/students', methods=['GET'])
@db_operation
def get_students():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM studentdetails")
    columns = [col[0] for col in cur.description]  # Get column names
    students = cur.fetchall()
    cur.close()

    # Convert to list of dictionaries for better JSON representation
    students_list = [dict(zip(columns, student)) for student in students]
    return jsonify(students_list)

@app.route('/api/add-marks', methods=['POST'])
@db_operation
def add_marks():
    data = request.json
    required_fields = ['prn', 'maths', 'science', 'english', 'history', 'geography']
    validate_input(data, required_fields)

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO marks_master 
        (prn, maths, science, english, history, geography) 
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        maths=%s, science=%s, english=%s, history=%s, geography=%s
    """, (
        data['prn'], data['maths'], data['science'], data['english'], 
        data['history'], data['geography'],
        data['maths'], data['science'], data['english'], 
        data['history'], data['geography']
    ))
    mysql.connection.commit()
    cur.close()

    return jsonify({'message': 'Marks added/updated successfully'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        validate_input(data, EXPECTED_FEATURES)
        features = prepare_features(data)

        # Make predictions
        rf_pred = float(rf_model.predict(features)[0])
        svm_pred = float(svm_model.predict(features)[0])
        nn_pred = float(nn_model.predict(np.array(features).reshape(1, -1))[0][0])

        return jsonify({
            'RandomForest': rf_pred,
            'SVM': svm_pred,
            'NeuralNetwork': nn_pred,
            'Average': (rf_pred + svm_pred + nn_pred) / 3
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': all([
            'rf_model' in globals(),
            'svm_model' in globals(),
            'scaler' in globals(),
            'nn_model' in globals()
        ])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)