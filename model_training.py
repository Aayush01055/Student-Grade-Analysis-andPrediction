import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = Path('/workspaces/ML-Project/cleaned_student_data.csv')
MODEL_DIR = Path('./models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load Data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    
    # Standardize column names: lowercase, replace spaces/hyphens, clean underscores
    df.columns = (
        df.columns.str.lower()
        .str.replace(r'[\s\-()\/]', '_', regex=True)  # Replace spaces, hyphens, slashes
        .str.replace('__', '_', regex=True)  # Remove double underscores
        .str.strip('_')  # Remove trailing underscores
    )
    
    logger.info(f"Dataset columns: {list(df.columns)}")
    return df

# Feature Engineering
def prepare_features(df):
    expected_features = [
        'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
        'lca_1_practical_performance', 'lca_2_active_learning_project',
        'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca'
    ]
    
    # Identify actual features in dataset after cleaning column names
    actual_features = [col for col in df.columns if any(feat in col for feat in expected_features)]
    
    if not actual_features:
        logger.error("No expected features found in dataset!")
        raise KeyError(f"Feature columns are missing from the dataset. Available columns: {df.columns}")
    
    logger.info(f"Using features: {actual_features}")

    # Extract target variable
    if 'overall_score' not in df.columns:
        logger.error("Target column 'overall_score' is missing!")
        raise KeyError("Target column 'overall_score' is missing.")
    
    X = df[actual_features]
    y = df['overall_score']
    
    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = MinMaxScaler().fit_transform(y.values.reshape(-1, 1)).flatten()
    
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_scaled, y_scaled

# Train SVM and RF
def train_models(X_train, y_train):
    svm_model = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    return svm_model, rf_model

# Evaluate Model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

# Main
def main():
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm_model, rf_model = train_models(X_train, y_train)
    
    with open(MODEL_DIR / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open(MODEL_DIR / 'rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    evaluate_model(svm_model, X_test, y_test, "SVM")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    logger.info("Model training and evaluation complete!")

if __name__ == '__main__':
    main()
