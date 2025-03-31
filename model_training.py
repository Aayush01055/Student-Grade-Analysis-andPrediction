import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor

# Configuration (use relative paths)
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'final_student_data.csv'
RF_MODEL_PATH = BASE_DIR / 'model/rf_student_performance_model.pkl'
XGB_MODEL_PATH = BASE_DIR / 'model/xgb_student_performance_model.pkl'
META_MODEL_PATH = BASE_DIR / 'model/meta_model.pkl'
SCALER_PATH = BASE_DIR / 'model/scaler.pkl'
TARGET_SCALER_PATH = BASE_DIR / 'model/target_scaler.pkl'
RESULTS_DIR = BASE_DIR / 'model/model_results'

# Create directories if they don’t exist
RF_MODEL_PATH.parent.mkdir(exist_ok=True)
XGB_MODEL_PATH.parent.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

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
    """Load and preprocess the data"""
    try:
        df = pd.read_csv(DATA_PATH)
        print("Data loaded. Shape:", df.shape)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please ensure final_student_data.csv exists.")

    # Clean column names
    df.columns = [clean_column_name(col) for col in df.columns]
    print("Cleaned columns:", df.columns.tolist())

    # Check for required columns
    required_columns = [
        'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
        'lca_1_practical_performance', 'lca_2_active_learning_project', 
        'lca_3_end_term_practical_oral', 'overall_score'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    # Ensure numeric data
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove outliers using IQR
    for col in required_columns[:-1]:  # Exclude overall_score
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

    # Drop rows with NaN values in required columns
    df = df.dropna(subset=required_columns)
    print("Shape after outlier removal and dropping NaN:", df.shape)

    return df

def prepare_features(df):
    """Prepare features from the dataset"""
    # Add derived features
    df['avg_cca'] = df[['cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks']].mean(axis=1)
    df['avg_lca'] = df[['lca_1_practical_performance', 'lca_2_active_learning_project', 'lca_3_end_term_practical_oral']].mean(axis=1)
    
    # Check if CO features exist, if not, fill with zeros
    for col in ['co1', 'co2', 'co3', 'co4']:
        if col not in df.columns:
            df[col] = 0.0
    df['avg_co'] = df[['co1', 'co2', 'co3', 'co4']].mean(axis=1)

    # Features to use
    feature_cols = [
        'cca_1_10_marks', 'cca_2_5_marks', 'cca_3_mid_term_15_marks',
        'lca_1_practical_performance', 'lca_2_active_learning_project',
        'lca_3_end_term_practical_oral', 'avg_cca', 'avg_lca',
        'co1', 'co2', 'co3', 'co4', 'avg_co'
    ]
    
    # Create feature matrix
    X = df[feature_cols]
    y = df['overall_score']
    
    print("\nFeatures being used:")
    print(X.columns.tolist())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale target variable (overall_score) to 0-100 range
    max_possible_score = 50  # Adjust based on max possible sum of CCA and LCA marks
    y_scaled = (y / max_possible_score) * 100
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_scaled.values.reshape(-1, 1)).flatten()
    
    # Save scalers
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(TARGET_SCALER_PATH, 'wb') as f:
        pickle.dump(target_scaler, f)
    
    return X_scaled, y_scaled, X.columns.tolist()

def train_rf_model(X_train, y_train):
    """Train and return a Random Forest regressor"""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print("Random Forest Cross-validation R² scores:", cv_scores)
    print("Random Forest Average CV R²:", cv_scores.mean())
    
    model.fit(X_train, y_train)
    return model

def train_xgb_model(X_train, y_train):
    """Train and return an XGBoost regressor with hyperparameter tuning"""
    model = XGBRegressor(random_state=42, objective='reg:squarederror')
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print("XGBoost Best parameters:", grid_search.best_params_)
    print("XGBoost Best cross-validation R² score:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    print("XGBoost Cross-validation R² scores:", cv_scores)
    print("XGBoost Average CV R²:", cv_scores.mean())
    
    return best_model

def train_meta_model(X_train, y_train, rf_model, xgb_model):
    """Train and return a Stacking Regressor as the meta-model"""
    estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model)
    ]
    meta_model = StackingRegressor(
        estimators=estimators,
        final_estimator=MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        cv=5
    )
    cv_scores = cross_val_score(meta_model, X_train, y_train, cv=5, scoring='r2')
    print("Meta-Model Cross-validation R² scores:", cv_scores)
    print("Meta-Model Average CV R²:", cv_scores.mean())
    
    meta_model.fit(X_train, y_train)
    return meta_model

def evaluate_models(rf_model, xgb_model, meta_model, X_test, y_test, feature_names, target_scaler):
    """Generate evaluation metrics and plots for RF, XGB, and Meta models"""
    # Individual predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_meta = meta_model.predict(X_test)
    
    # Inverse transform predictions to original scale
    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_rf_unscaled = target_scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()
    y_pred_xgb_unscaled = target_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()
    y_pred_meta_unscaled = target_scaler.inverse_transform(y_pred_meta.reshape(-1, 1)).flatten()
    
    # Metrics for Random Forest
    print("\nRandom Forest Metrics on Test Set:")
    mse_rf = mean_squared_error(y_test_unscaled, y_pred_rf_unscaled)
    mae_rf = mean_absolute_error(y_test_unscaled, y_pred_rf_unscaled)
    r2_rf = r2_score(y_test_unscaled, y_pred_rf_unscaled)
    print(f"Mean Squared Error (MSE): {mse_rf:.3f}")
    print(f"Mean Absolute Error (MAE): {mae_rf:.3f}")
    print(f"R² Score: {r2_rf:.3f}")
    
    # Metrics for XGBoost
    print("\nXGBoost Metrics on Test Set:")
    mse_xgb = mean_squared_error(y_test_unscaled, y_pred_xgb_unscaled)
    mae_xgb = mean_absolute_error(y_test_unscaled, y_pred_xgb_unscaled)
    r2_xgb = r2_score(y_test_unscaled, y_pred_xgb_unscaled)
    print(f"Mean Squared Error (MSE): {mse_xgb:.3f}")
    print(f"Mean Absolute Error (MAE): {mae_xgb:.3f}")
    print(f"R² Score: {r2_xgb:.3f}")
    
    # Metrics for Meta-Model
    print("\nMeta-Model (Stacking) Metrics on Test Set:")
    mse_meta = mean_squared_error(y_test_unscaled, y_pred_meta_unscaled)
    mae_meta = mean_absolute_error(y_test_unscaled, y_pred_meta_unscaled)
    r2_meta = r2_score(y_test_unscaled, y_pred_meta_unscaled)
    print(f"Mean Squared Error (MSE): {mse_meta:.3f}")
    print(f"Mean Absolute Error (MAE): {mae_meta:.3f}")
    print(f"R² Score: {r2_meta:.3f}")
    
    # Scatter plot of actual vs predicted (Meta-Model)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_unscaled, y_pred_meta_unscaled, alpha=0.5, label='Meta-Model')
    plt.plot([y_test_unscaled.min(), y_test_unscaled.max()], [y_test_unscaled.min(), y_test_unscaled.max()], 'r--', lw=2)
    plt.xlabel('Actual Overall Score')
    plt.ylabel('Predicted Overall Score')
    plt.title('Actual vs Predicted Overall Scores (Meta-Model)')
    plt.legend()
    plt.savefig(RESULTS_DIR / 'actual_vs_predicted_meta.png')
    plt.close()
    
    return {'mse': mse_meta, 'mae': mae_meta, 'r2': r2_meta}

def feature_importance_analysis(rf_model, xgb_model, feature_names):
    """Analyze and visualize feature importance for RF and XGB models"""
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    xgb_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    rf_importance['Feature'] = rf_importance['Feature'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    xgb_importance['Feature'] = xgb_importance['Feature'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=rf_importance)
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'rf_feature_importance.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=xgb_importance)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'xgb_feature_importance.png')
    plt.close()
    
    return rf_importance, xgb_importance

def save_results(rf_model, xgb_model, meta_model, metrics):
    """Save models and results to files"""
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open(META_MODEL_PATH, 'wb') as f:
        pickle.dump(meta_model, f)
    
    with open(RESULTS_DIR / 'model_performance.txt', 'w') as f:
        f.write("=== Meta-Model Performance ===\n")
        f.write(f"Mean Squared Error (MSE): {metrics['mse']:.3f}\n")
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.3f}\n")
        f.write(f"R² Score: {metrics['r2']:.3f}\n")
    
    print(f"\nResults saved to {RESULTS_DIR}")

def main():
    try:
        print("\nStarting model training...")
        
        # Load and prepare data
        df = load_data()
        X, y, feature_names = prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Train base models
        rf_model = train_rf_model(X_train, y_train)
        xgb_model = train_xgb_model(X_train, y_train)
        
        # Train meta-model
        meta_model = train_meta_model(X_train, y_train, rf_model, xgb_model)
        
        # Load target scaler for evaluation
        with open(TARGET_SCALER_PATH, 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Evaluate models
        metrics = evaluate_models(rf_model, xgb_model, meta_model, X_test, y_test, feature_names, target_scaler)
        
        # Feature analysis (for base models only)
        rf_importance, xgb_importance = feature_importance_analysis(rf_model, xgb_model, feature_names)
        print("\nRandom Forest Feature Importance:")
        print(rf_importance)
        print("\nXGBoost Feature Importance:")
        print(xgb_importance)
        
        # Save results
        save_results(rf_model, xgb_model, meta_model, metrics)
        
        print("\nTraining complete!")
        print("Key files generated:")
        print(f"- Random Forest Model: {RF_MODEL_PATH}")
        print(f"- XGBoost Model: {XGB_MODEL_PATH}")
        print(f"- Meta Model: {META_MODEL_PATH}")
        print(f"- Scaler: {SCALER_PATH}")
        print(f"- Target Scaler: {TARGET_SCALER_PATH}")
        print(f"- Results directory: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"\nError during model training: {str(e)}")

if __name__ == '__main__':
    main()