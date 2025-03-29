# visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuration
DATA_PATH = Path('C:/Users/samik/Downloads/ML/ML/cleaned_student_data.csv')
PLOTS_DIR = Path('eda_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def clean_column_names(df):
    """Standardize column names"""
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[\(\)]', '', regex=True)
    return df

def load_data():
    """Load and clean data"""
    df = pd.read_csv(DATA_PATH)
    print("Data loaded. Shape:", df.shape)
    df = clean_column_names(df)
    print("Cleaned columns:", df.columns.tolist())
    return df

def basic_summary(df):
    """Generate basic statistics"""
    print("\n=== BASIC STATISTICS ===")
    print(df.describe(include='all'))
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    print("\nSkewness:\n", df[numeric_cols].skew())
    print("\nKurtosis:\n", df[numeric_cols].kurt())

def plot_distributions(df):
    """Plot distributions of key variables"""
    print("\nPlotting distributions...")
    try:
        # Overall Score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Overall_Score'], bins=20, kde=True)
        plt.title('Distribution of Overall Scores')
        plt.savefig(PLOTS_DIR/'overall_score_dist.png')
        plt.close()
        
        # Individual component distributions
        components = ['CCA-1_10_marks', 'CCA-2_5_marks', 'LCA-1_PRACTICAL_PERFORMANCE']
        for col in components:
            if col in df.columns:
                plt.figure(figsize=(8, 5))
                sns.histplot(df[col], bins=15, kde=True)
                plt.title(f'Distribution of {col}')
                plt.savefig(PLOTS_DIR/f'{col}_dist.png')
                plt.close()
                
    except Exception as e:
        print(f"Error plotting distributions: {str(e)}")

def plot_correlations(df):
    """Plot correlation matrix"""
    print("\nAnalyzing correlations...")
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig(PLOTS_DIR/'correlation_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")

def main():
    try:
        print("Starting EDA...")
        df = load_data()
        
        # Create Overall Score if not exists
        if 'Overall_Score' not in df.columns:
            df['Overall_Score'] = df['Overall_Score'] if 'Overall_Score' in df.columns else df['Overall_Score']
        
        basic_summary(df)
        plot_distributions(df)
        plot_correlations(df)
        
        print("\nEDA complete! Check plots in:", PLOTS_DIR)
    except Exception as e:
        print(f"\nError during EDA: {str(e)}")

if __name__ == '__main__':
    main()