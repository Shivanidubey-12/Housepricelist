```python
import os
import pandas as pd
import xgboost as xgb
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)


# Load trained model
@st.cache_resource
def load_model():

    # Get the folder where this Python file is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Model file path
    model_path = os.path.join(
        base_dir,
        "xgb_house_model.json"
    )

    # Check whether model exists
    if not os.path.exists(model_path):
        st.error("❌ xgb_house_model.json file not found!")
        st.info(
            "Please keep xgb_house_model.json "
            "in the same folder as House_price_prediction.py."
        )
        st.stop()

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    return model


def main():

    st.title("🏠 House Price Prediction")

    st.write(
        "This app will help you predict the house price."
    )

    # Load model
    model = load_model()

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

    # Create DataFrame
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
    if st.button("🔮 Predict House Price"):

        try:

            prediction = model.predict(data_new)

            st.success(
                "Estimated House Price: ₹ {:.2f} Lakhs".format(
                    prediction[0]
                )
            )

        except Exception as e:

            st.error("❌ Prediction failed.")
            st.error(str(e))


if __name__ == "__main__":
    main()
```

### VERY IMPORTANT: Check your folder

Go to:

```text
C:\Users\dsaty\Desktop\ML project\Project 3. House_price_predication\
```

You need to have:

```text
Project 3. House_price_predication
│
├── House_price_prediction.py
├── xgb_house_model.json    ← MUST BE HERE
└── requirements.txt
```

### Check using PowerShell

Run:

```powershell
cd "C:\Users\dsaty\Desktop\ML project\Project 3. House_price_predication"
```

Then:

```powershell
dir
```

You should see:

```text
House_price_prediction.py
xgb_house_model.json
requirements.txt
```

If you **don't see `xgb_house_model.json`**, that's exactly why you're getting this error.

Then run:

```powershell
streamlit run House_price_prediction.py
```

### For GitHub/Streamlit deployment

Your GitHub repository should also have:

```text
House_price_prediction.py
xgb_house_model.json
requirements.txt
```

The model filename must be **exactly**:

```text
xgb_house_model.json
```

Not:

```text
xgb_house_model (1).json
xgb_house_model.json.json
XGB_house_model.json
xgb_house_model.JSON
```

Your original code confirms that the model is expected to be loaded from `xgb_house_model.json`.

**If you have the model file somewhere else on your laptop, send/upload `xgb_house_model.json` here. I can check it and tell you exactly where to put it and whether your model is compatible with this code.**
