from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import pickle
import numpy as np
import tensorflow as tf
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'student_db'

mysql = MySQL(app)

# Load trained models
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

nn_model = tf.keras.models.load_model('models/nn_model.h5')

def prepare_features(data):
    expected_features = ['cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
                         'lca_1_practical_performance', 'lca_2_active_learning_project',
                         'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca']
    features = [data[f] for f in expected_features]
    return scaler.transform([features])

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    prn = data.get('prn')
    password = data.get('password')

    cur = mysql.connection.cursor()
    cur.execute("SELECT password FROM loginmaster WHERE prn=%s", (prn,))
    user = cur.fetchone()
    cur.close()

    if user and check_password_hash(user[0], password):
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/add-user', methods=['POST'])
def add_user():
    data = request.json
    prn = data['prn']
    password = generate_password_hash(data['password'])

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO loginmaster (prn, password) VALUES (%s, %s)", (prn, password))
    mysql.connection.commit()
    cur.close()

    return jsonify({'message': 'User registered successfully'})

@app.route('/api/students', methods=['GET'])
def get_students():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM studentdetails")
    students = cur.fetchall()
    cur.close()

    return jsonify(students)

@app.route('/api/add-marks', methods=['POST'])
def add_marks():
    data = request.json

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO marks_master (prn, maths, science, english, history, geography) VALUES (%s, %s, %s, %s, %s, %s)",
                (data['prn'], data['maths'], data['science'], data['english'], data['history'], data['geography']))
    mysql.connection.commit()
    cur.close()

    return jsonify({'message': 'Marks added successfully'})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    features = prepare_features(data)

    rf_pred = rf_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]
    nn_pred = nn_model.predict(np.array(features).reshape(1, -1))[0][0]

    average_pred = (rf_pred + svm_pred + nn_pred) / 3

    return jsonify({'RandomForest': rf_pred, 'SVM': svm_pred, 'NeuralNetwork': nn_pred, 'Average': average_pred})

if __name__ == '__main__':
    app.run(debug=True)
