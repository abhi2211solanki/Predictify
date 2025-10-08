import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_data, preprocess_data

def plot_feature_importance():
    # Load data and preprocess it (same as training)
    data = load_data("data/train.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Load the trained model
    model = joblib.load("src/house_price_model.pkl")

    # Automatically extract numeric column names (used during training)
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.drop(["Id", "SalePrice"], errors="ignore")

    # Match lengths
    importance = model.coef_
    if len(numeric_cols) != len(importance):
        print(f"⚠️ Mismatch detected: {len(numeric_cols)} features vs {len(importance)} coefficients.")
        # Adjust to smallest length
        min_len = min(len(numeric_cols), len(importance))
        numeric_cols = numeric_cols[:min_len]
        importance = importance[:min_len]

    # Create dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Coefficient': importance
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print("\n📊 Top 10 Most Influential Features:")
    print(importance_df.head(10))

    # Plot top 10
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:10], importance_df['Coefficient'][:10], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 Feature Importances (Linear Regression Coefficients)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_feature_importance()
