import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file_path):
    """Load and preprocess student grade data."""
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = df.columns.str.strip()

    print(f"Raw Data Loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Define column groups
    co_columns = [
        "CO1 ( Analyze and apply different data preparation techniques for Machine Learning applications )",
        "CO2 ( Identify, Analyze and compare appropriate supervised learning algorithm for given problem )",
        "CO3 (Identify, Analyze and Compare Unsupervised and semi supervised algorithms )",
        "CO4(Design and implement Machine Learning techniques for real-time applications )"
    ]
    
    cca_columns = ["CCA-1 (10 marks)", "CCA-2 (5 marks)", "CCA-3  (Mid term-15 marks)"]
    lca_columns = ["LCA 1 (PRACTICAL PERFORMANCE)", "LCA-2 (Active Learning/ Project)", "LCA-3 (End term practical/oral)"]

    # Convert categorical CO values ('Yes'/'No') to numeric
    for col in co_columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Convert CCA & LCA to numeric
    for col in cca_columns + lca_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    df[cca_columns] = df[cca_columns].fillna(df[cca_columns].mean())  # Mean Imputation
    df[lca_columns] = df[lca_columns].fillna(df[lca_columns].median())  # Median Imputation

    # Drop rows with remaining NaNs
    df.dropna(inplace=True)

    # Compute Overall Score (Weighted Sum)
    weights = {
        "CCA-1 (10 marks)": 0.2, "CCA-2 (5 marks)": 0.1, "CCA-3  (Mid term-15 marks)": 0.3,
        "LCA 1 (PRACTICAL PERFORMANCE)": 0.15, "LCA-2 (Active Learning/ Project)": 0.15, "LCA-3 (End term practical/oral)": 0.2
    }
    #Calculate avg_cca,avg_lca
    df["Overall Score"] = sum(df[col] * weight for col, weight in weights.items())

    print(f"Cleaned Data Shape: {df.shape}")
    
    # Save cleaned data
    cleaned_file_path = "C:/Users/samik/Downloads/ML/ML/cleaned_student_data.csv"
    df.to_csv(cleaned_file_path, index=False)
    
    print(f"Cleaned data saved to: {cleaned_file_path}")

    # Plot Score Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Overall Score"], bins=20, kde=True, color="blue")
    plt.title("Distribution of Overall Scores")
    plt.xlabel("Overall Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "C:/Users/samik/Downloads/ML/ML/data/Updated_Student_Grade_Indian_Names_Email.csv"  # Change path if needed
    preprocess_data(file_path)
