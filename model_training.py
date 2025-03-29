import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration (use absolute paths for clarity)
BASE_DIR = Path(__file__).parent
DATA_PATH = Path('C:/Users/samik/Downloads/ML/ML/cleaned_student_data.csv')  # Updated to match preprocessing output
MODEL_DIR = Path('C:/Users/samik/Downloads/ML/ML/models')
RESULTS_DIR = Path('C:/Users/samik/Downloads/ML/ML/model_results')
RF_MODEL_PATH = MODEL_DIR / 'rf_student_performance_model.pkl'
XGB_MODEL_PATH = MODEL_DIR / 'xgb_student_performance_model.pkl'
NN_MODEL_PATH = MODEL_DIR / 'nn_student_performance_model.h5'
META_MODEL_PATH = MODEL_DIR / 'meta_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
TARGET_SCALER_PATH = MODEL_DIR / 'target_scaler.pkl'

# Create directories
for path in [MODEL_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def clean_column_name(col):
    """Clean and standardize column names"""
    return (col.strip()
            .lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('-', '_')
            .replace('/', '_')
            .replace(',', '')
            .replace('__', '_'))

def load_data():
    """Load and validate the data"""
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {DATA_PATH}")
        raise

    df.columns = [clean_column_name(col) for col in df.columns]
    logger.info(f"Cleaned columns: {df.columns.tolist()}")

    required_columns = [
        'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
        'lca_1_practical_performance', 'lca_2_active_learning_project',
        'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca', 'overall_score'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required_columns)
    logger.info(f"Shape after dropping NaN: {df.shape}")
    return df

def prepare_features(df):
    """Prepare features and target with proper scaling"""
    feature_cols = [
        'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
        'lca_1_practical_performance', 'lca_2_active_learning_project',
        'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca'
    ]
    X = df[feature_cols]
    y = df['overall_score']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    target_scaler = MinMaxScaler(feature_range=(0, 1))  # Scale target to 0-1
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(TARGET_SCALER_PATH, 'wb') as f:
        pickle.dump(target_scaler, f)

    logger.info(f"Features prepared: {feature_cols}")
    return X_scaled, y_scaled, feature_cols

def train_rf_model(X_train, y_train):
    """Train Random Forest with hyperparameter tuning"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best RF params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_xgb_model(X_train, y_train):
    """Train XGBoost with hyperparameter tuning"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best XGB params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_nn_model(X_train, y_train):
    """Train Neural Network with early stopping"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                       validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(RESULTS_DIR / 'nn_training_history.png')
    plt.close()
    
    return model

def train_meta_model(X_train, y_train, rf_model, xgb_model, nn_model):
    """Train a meta-model with cross-validated stacking"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_preds = np.zeros(len(X_train))
    xgb_preds = np.zeros(len(X_train))
    nn_preds = np.zeros(len(X_train))
    
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold = y_train[train_idx]
        
        rf_model.fit(X_train_fold, y_train_fold)
        xgb_model.fit(X_train_fold, y_train_fold)
        nn_model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, 
                    validation_split=0.2, verbose=0)
        
        rf_preds[val_idx] = rf_model.predict(X_val_fold)
        xgb_preds[val_idx] = xgb_model.predict(X_val_fold)
        nn_preds[val_idx] = nn_model.predict(X_val_fold, verbose=0).flatten()
    
    stacked_X = np.column_stack((rf_preds, xgb_preds, nn_preds))
    meta_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    meta_model.fit(stacked_X, y_train)
    
    logger.info("Meta-model trained successfully")
    return meta_model

def evaluate_models(rf_model, xgb_model, nn_model, meta_model, X_test, y_test, feature_names, target_scaler):
    """Evaluate all models with comprehensive metrics and plots"""
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_nn = nn_model.predict(X_test, verbose=0).flatten()
    stacked_X_test = np.column_stack((y_pred_rf, y_pred_xgb, y_pred_nn))
    y_pred_stacked = meta_model.predict(stacked_X_test)
    
    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions = {
        'rf': target_scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten(),
        'xgb': target_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten(),
        'nn': target_scaler.inverse_transform(y_pred_nn.reshape(-1, 1)).flatten(),
        'stacked': target_scaler.inverse_transform(y_pred_stacked.reshape(-1, 1)).flatten()
    }
    
    metrics_dict = {}
    for name, y_pred in predictions.items():
        mse = mean_squared_error(y_test_unscaled, y_pred)
        mae = mean_absolute_error(y_test_unscaled, y_pred)
        r2 = r2_score(y_test_unscaled, y_pred)
        metrics_dict[name] = {'mse': mse, 'mae': mae, 'r2': r2}
        logger.info(f"{name.upper()} Metrics: MSE={mse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
    
    # Plot predictions
    plt.figure(figsize=(12, 8))
    for name, y_pred in predictions.items():
        plt.scatter(y_test_unscaled, y_pred, alpha=0.5, label=name.upper())
    plt.plot([y_test_unscaled.min(), y_test_unscaled.max()], 
             [y_test_unscaled.min(), y_test_unscaled.max()], 'r--', lw=2)
    plt.xlabel('Actual Overall Score')
    plt.ylabel('Predicted Overall Score')
    plt.title('Actual vs Predicted Scores (All Models)')
    plt.legend()
    plt.savefig(RESULTS_DIR / 'actual_vs_predicted_all.png')
    plt.close()
    
    return metrics_dict

def feature_importance_analysis(rf_model, xgb_model, feature_names):
    """Analyze and visualize feature importance"""
    rf_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
    xgb_importance = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_model.feature_importances_}).sort_values('Importance', ascending=False)
    
    for df in [rf_importance, xgb_importance]:
        df['Feature'] = df['Feature'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=rf_importance)
    plt.title('Random Forest Feature Importance')
    plt.savefig(RESULTS_DIR / 'rf_feature_importance.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=xgb_importance)
    plt.title('XGBoost Feature Importance')
    plt.savefig(RESULTS_DIR / 'xgb_feature_importance.png')
    plt.close()
    
    return rf_importance, xgb_importance

def save_results(rf_model, xgb_model, nn_model, meta_model, metrics_dict):
    """Save models and results"""
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_model, f)
    nn_model.save(NN_MODEL_PATH)
    with open(META_MODEL_PATH, 'wb') as f:
        pickle.dump(meta_model, f)
    
    with open(RESULTS_DIR / 'model_performance.txt', 'w') as f:
        for model_name, metrics in metrics_dict.items():
            f.write(f"\n=== {model_name.upper()} Model Performance ===\n")
            f.write(f"MSE: {metrics['mse']:.3f}\n")
            f.write(f"MAE: {metrics['mae']:.3f}\n")
            f.write(f"R²: {metrics['r2']:.3f}\n")
    
    logger.info(f"Results saved to {RESULTS_DIR}")

def main():
    try:
        logger.info("Starting model training...")
        
        df = load_data()
        X, y, feature_names = prepare_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = train_rf_model(X_train, y_train)
        xgb_model = train_xgb_model(X_train, y_train)
        nn_model = train_nn_model(X_train, y_train)
        meta_model = train_meta_model(X_train, y_train, rf_model, xgb_model, nn_model)
        
        with open(TARGET_SCALER_PATH, 'rb') as f:
            target_scaler = pickle.load(f)
        
        metrics_dict = evaluate_models(rf_model, xgb_model, nn_model, meta_model, 
                                     X_test, y_test, feature_names, target_scaler)
        
        rf_importance, xgb_importance = feature_importance_analysis(rf_model, xgb_model, feature_names)
        logger.info("\nRandom Forest Feature Importance:\n" + rf_importance.to_string())
        logger.info("\nXGBoost Feature Importance:\n" + xgb_importance.to_string())
        
        save_results(rf_model, xgb_model, nn_model, meta_model, metrics_dict)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == '__main__':
    main()