import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import load_data, preprocess_data

def train_model():
    # Load and preprocess data
    data = load_data("data/train.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Save model and scaler for future use (optional)
    import joblib
    joblib.dump(model, "src/house_price_model.pkl")
    joblib.dump(scaler, "src/scaler.pkl")

    print("\n💾 Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model()
