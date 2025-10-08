import joblib
import numpy as np

def predict_price(area, bedrooms, bathrooms, stories, parking):
    # Load the trained model and scaler
    model = joblib.load("src/house_price_model.pkl")
    scaler = joblib.load("src/scaler.pkl")

    # Prepare input data
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    input_scaled = scaler.transform(input_data)

    # Predict
    predicted_price = model.predict(input_scaled)
    print(f"🏡 Predicted House Price: ₹{predicted_price[0]:,.2f}")

if __name__ == "__main__":
    # Example: You can change these values to test
    predict_price(area=3200, bedrooms=4, bathrooms=3, stories=2, parking=2)
