import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import os


def main():

    # Page title
    st.title(" House Price Prediction")

    st.write("This app will help you predict the house price.")

    # Load trained model
    model = xgb.XGBRegressor()

    # Get the folder where this Python file is located
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "xgb_model.json"
    )

    model.load_model(model_path)

    # User inputs
    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=50
    )

    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=100,
        max_value=100000,
        value=5000,
        step=100
    )

    floors = st.number_input(
        "Number of Floors",
        min_value=1.0,
        max_value=4.0,
        value=1.0,
        step=0.5
    )

    waterfront = st.selectbox(
        "Waterfront",
        ("No", "Yes")
    )

    if waterfront == "No":
        waterfront_value = 0
    else:
        waterfront_value = 1

    condition = st.slider(
        "House Condition",
        min_value=1,
        max_value=5,
        value=3
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2000,
        step=1
    )

    # Calculate house age
    current_year = 2026
    age = current_year - year_built

    # Create dataframe
    data_new = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "living_area": [living_area],
        "lot_area": [lot_area],
        "floors": [floors],
        "waterfront": [waterfront_value],
        "condition": [condition],
        "age": [age]
    })

    # Prediction
    if st.button("Predict House Price"):

        prediction = model.predict(data_new)

        st.success(
            "Estimated House Price: ₹ {:.2f} Lakhs".format(prediction[0])
        )

    # Project URL
    st.markdown("---")
    st.markdown(
        "🔗 **Project:** [House Price Prediction App](https://streamlit.io/)"
    )


if __name__ == "__main__":
    main()
