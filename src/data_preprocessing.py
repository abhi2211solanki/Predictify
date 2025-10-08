import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data = pd.read_csv(path)
    print("✅ Data loaded successfully!")
    print(data.head())
    print("\nColumns:", data.columns.tolist())
    return data


def preprocess_data(data):
    # Select essential features (same 6 used in app)
    selected_features = [
        'OverallQual',   # Quality
        'GrLivArea',     # Area
        'GarageCars',    # Garage capacity
        'TotalBsmtSF',   # Basement size
        'FullBath',      # Bathrooms
        'YearBuilt'      # Year built
    ]

    # Filter the data to only these columns
    X = data[selected_features]
    y = data["SalePrice"]

    # Fill missing numeric values with mean
    X = X.fillna(X.mean())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"✅ Preprocessing complete! Using {len(selected_features)} features: {selected_features}")
    return X_train, X_test, y_train, y_test, scaler



if __name__ == "__main__":
    data = load_data("data/train.csv")
    preprocess_data(data)
