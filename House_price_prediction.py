```python
import os
import pandas as pd
import xgboost as xgb
import streamlit as st


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "xgb_house_model.json"
    )

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    return model


# -----------------------------
# Main App
# -----------------------------
def main():

    st.title("🏠 House Price Prediction")

    st.write(
        "Enter the house details below to predict the estimated house price."
    )

    # Load trained model
    try:
        model = load_model()
    except Exception as e:
        st.error("Model file could not be loaded.")
        st.error(f"Error: {e}")
        st.info(
            "Make sure xgb_house_model.json is uploaded "
            "to the same GitHub folder as this Python file."
        )
        st.stop()

    # -----------------------------
    # User Inputs
    # -----------------------------

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

    waterfront_value = 0 if waterfront == "No" else 1

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

    # -----------------------------
    # Create Input DataFrame
    # -----------------------------

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

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("🔮 Predict House Price", use_container_width=True):

        try:
            prediction = model.predict(data_new)

            price = prediction[0]

            st.success(
                f"Estimated House Price: ₹ {price:.2f} Lakhs"
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.error(f"Error: {e}")


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":
    main()
```

### Your GitHub folder should be

```text
House-Price-Prediction
│
├── House_price_prediction.py
├── xgb_house_model.json
├── requirements.txt
└── README.md
```

### `requirements.txt`

```text
streamlit
pandas
xgboost
```

You don't actually use NumPy directly in the revised code, so I've removed it from the requirements.

### Before deploying, test it

In PowerShell:

```powershell
cd "C:\Users\dsaty\Desktop\ML project"
```

Install:

```powershell
pip install -r requirements.txt
```

Run:

```powershell
streamlit run House_price_prediction.py
```

If your file is actually named `House_price_predication.py` (with **predication**), use:

```powershell
streamlit run House_price_predication.py
```

**Important:** Keep the Python filename and the Streamlit deployment **Main file path** exactly the same.


