import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Predictify 🏠", page_icon="🏡", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("src/house_price_model.pkl")
scaler = joblib.load("src/scaler.pkl")

# Load dataset for charts
data = pd.read_csv("data/train.csv")

st.title("🏠 Predictify - House Price Prediction Dashboard")
st.markdown("### Enter house details below to predict the estimated price and explore key insights 📊")

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["💰 Prediction", "📊 Data Insights"])

# ==================================================================
# TAB 1 : PREDICTION
# ==================================================================
with tab1:
    st.subheader("💰 Make a Price Prediction")

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10, step=1)
        fullbath = st.number_input("Full Bathrooms", min_value=1, max_value=5, step=1)
    with col2:
        area = st.number_input("Area (GrLivArea in sq ft)", min_value=500, max_value=10000, step=100)
        total_bsmt = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, step=100)
    with col3:
        garage_cars = st.number_input("Garage Capacity (cars)", min_value=0, max_value=5, step=1)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2024, step=1)

    if st.button("Predict Price 💸"):
        # Prepare input
        input_data = np.array([[overall_qual, area, garage_cars, total_bsmt, fullbath, year_built]])
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]

        # Display predicted price
        st.success(f"🏡 Estimated House Price: ₹{predicted_price:,.2f}")

        # ---------------- PIE CHART COMPARISON ----------------
        avg_price = data["SalePrice"].mean()
        ratio = predicted_price / avg_price

        st.markdown("---")
        st.markdown("### 🥧 Market Comparison")

        labels = ["Your House", "Average Market Price"]
        values = [predicted_price, avg_price]

        fig3 = px.pie(
            names=labels,
            values=values,
            color_discrete_sequence=["#4CAF50", "#FFC107"],
            title="Your House Value vs Market Average",
            hole=0.4
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Dynamic feedback text
        if ratio > 1.1:
            st.success(f"🟢 Your house is **{ratio:.2f}× above** the average market value! Premium property 💎")
        elif 0.9 <= ratio <= 1.1:
            st.info(f"🟡 Your house is roughly **average** in market value ({ratio:.2f}×).")
        else:
            st.warning(f"🔴 Your house is **{ratio:.2f}× below** the average market value. Potential budget-friendly property.")

        # ---------------- SCATTER PLOT: Your House vs Market ----------------
        st.markdown("---")
        st.markdown("### 📈 Your House Position in Market Trend")

        # Create scatter plot (Area vs Price)
        fig4 = px.scatter(
            data,
            x="GrLivArea",
            y="SalePrice",
            color_discrete_sequence=["#3498db"],
            opacity=0.6,
            title="Living Area vs Sale Price (Market Distribution)",
            labels={"GrLivArea": "Living Area (sq ft)", "SalePrice": "Price (₹)"}
        )

        # Add predicted point
        fig4.add_scatter(
            x=[area],
            y=[predicted_price],
            mode="markers+text",
            marker=dict(color="red", size=12, symbol="star"),
            text=["Your House"],
            textposition="top center",
            name="Your House"
        )

        st.plotly_chart(fig4, use_container_width=True)
        st.caption("🔹 Blue dots show real market data; the red star marks your predicted property.")

# ==================================================================
# TAB 2 : VISUALIZATIONS
# ==================================================================
with tab2:
    st.subheader("📊 Data Insights & Visualizations")

    # ---------------- Price Distribution ----------------
    st.markdown("#### 1️⃣ Distribution of House Prices")
    fig1, ax1 = plt.subplots()
    ax1.hist(data["SalePrice"], bins=30, color='skyblue', edgecolor='black')
    ax1.set_title("Distribution of House Prices")
    ax1.set_xlabel("Sale Price")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # ---------------- Feature Importance ----------------
    st.markdown("#### 2️⃣ Top 10 Most Important Features")
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.drop(["Id", "SalePrice"], errors="ignore")
    importance = model.coef_

    min_len = min(len(numeric_cols), len(importance))
    feature_importance = pd.DataFrame({
        "Feature": numeric_cols[:min_len],
        "Coefficient": importance[:min_len]
    }).sort_values(by="Coefficient", key=abs, ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    ax2.barh(feature_importance["Feature"], feature_importance["Coefficient"], color='salmon')
    ax2.set_title("Top 10 Most Influential Features")
    ax2.set_xlabel("Coefficient Value")
    ax2.invert_yaxis()
    st.pyplot(fig2)

    st.caption("🔹 Larger coefficients indicate stronger influence on the predicted sale price.")
