import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# --------------------------------------------------
# Find the main Assignment1 folder
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------
model = joblib.load(BASE_DIR / "car_price_model.pkl")


# --------------------------------------------------
# Load the feature names used during training
# --------------------------------------------------
model_features = joblib.load(BASE_DIR / "model_features.pkl")


print("Model loaded successfully!")
print("Number of model features:", len(model_features))


# --------------------------------------------------
# Create Dash application
# --------------------------------------------------
app = dash.Dash(__name__)

app.title = "Car Price Prediction"


# --------------------------------------------------
# Application layout
# --------------------------------------------------
app.layout = html.Div(
    [

        html.H1("Car Price Prediction"),

        html.P(
            "Enter the car information below to predict its selling price."
        ),

        html.P(
            "You may leave fields blank. Missing numerical values will "
            "be replaced using reasonable default values."
        ),

        html.Hr(),

        html.H3("Car Information"),


        # --------------------------------------------------
        # Year
        # --------------------------------------------------
        html.Label("Year"),

        dcc.Input(
            id="year",
            type="number",
            placeholder="e.g. 2018"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Kilometers driven
        # --------------------------------------------------
        html.Label("Kilometers Driven"),

        dcc.Input(
            id="km_driven",
            type="number",
            placeholder="e.g. 50000"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Owner
        # --------------------------------------------------
        html.Label("Owner"),

        dcc.Dropdown(
            id="owner",
            options=[
                {"label": "First Owner", "value": 1},
                {"label": "Second Owner", "value": 2},
                {"label": "Third Owner", "value": 3},
                {"label": "Fourth & Above Owner", "value": 4},
            ],
            placeholder="Select owner"
        ),

        html.Br(),


        # --------------------------------------------------
        # Mileage
        # --------------------------------------------------
        html.Label("Mileage (kmpl)"),

        dcc.Input(
            id="mileage",
            type="number",
            placeholder="e.g. 20.5"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Engine
        # --------------------------------------------------
        html.Label("Engine (CC)"),

        dcc.Input(
            id="engine",
            type="number",
            placeholder="e.g. 1500"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Maximum Power
        # --------------------------------------------------
        html.Label("Max Power (bhp)"),

        dcc.Input(
            id="max_power",
            type="number",
            placeholder="e.g. 100"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Seats
        # --------------------------------------------------
        html.Label("Number of Seats"),

        dcc.Input(
            id="seats",
            type="number",
            placeholder="e.g. 5"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Fuel
        # --------------------------------------------------
        html.Label("Fuel"),

        dcc.Dropdown(
            id="fuel",
            options=[
                {"label": "Diesel", "value": "Diesel"},
                {"label": "Petrol", "value": "Petrol"},
            ],
            placeholder="Select fuel"
        ),

        html.Br(),


        # --------------------------------------------------
        # Seller Type
        # --------------------------------------------------
        html.Label("Seller Type"),

        dcc.Dropdown(
            id="seller_type",
            options=[
                {"label": "Individual", "value": "Individual"},
                {"label": "Dealer", "value": "Dealer"},
                {"label": "Trustmark Dealer", "value": "Trustmark Dealer"},
            ],
            placeholder="Select seller type"
        ),

        html.Br(),


        # --------------------------------------------------
        # Transmission
        # --------------------------------------------------
        html.Label("Transmission"),

        dcc.Dropdown(
            id="transmission",
            options=[
                {"label": "Manual", "value": "Manual"},
                {"label": "Automatic", "value": "Automatic"},
            ],
            placeholder="Select transmission"
        ),

        html.Br(),


        # --------------------------------------------------
        # Brand
        # --------------------------------------------------
        html.Label("Brand"),

        dcc.Input(
            id="brand",
            type="text",
            placeholder="e.g. Toyota"
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Prediction button
        # --------------------------------------------------
        html.Button(
            "Predict Selling Price",
            id="predict_button",
            n_clicks=0
        ),

        html.Br(),
        html.Br(),


        # --------------------------------------------------
        # Prediction result
        # --------------------------------------------------
        html.Div(
            id="prediction_output",
            style={
                "fontSize": "24px",
                "fontWeight": "bold"
            }
        ),

    ],

    style={
        "width": "60%",
        "margin": "auto",
        "padding": "30px"
    }
)


# ==========================================================
# CALLBACK
# ==========================================================

@app.callback(
    Output("prediction_output", "children"),

    Input("predict_button", "n_clicks"),

    State("year", "value"),
    State("km_driven", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("brand", "value"),
)
def predict_price(
    n_clicks,
    year,
    km_driven,
    owner,
    mileage,
    engine,
    max_power,
    seats,
    fuel,
    seller_type,
    transmission,
    brand
):

    # Do nothing before the button is clicked
    if not n_clicks:
        return ""

    try:

        # --------------------------------------------------
        # Imputation for missing numerical values
        # --------------------------------------------------
        year = 2018 if year is None else year
        km_driven = 50000 if km_driven is None else km_driven
        owner = 1 if owner is None else owner
        mileage = 20.0 if mileage is None else mileage
        engine = 1500 if engine is None else engine
        max_power = 100 if max_power is None else max_power
        seats = 5 if seats is None else seats


        # --------------------------------------------------
        # Imputation for missing categorical values
        # --------------------------------------------------
        fuel = "Petrol" if fuel is None else fuel
        seller_type = "Individual" if seller_type is None else seller_type
        transmission = "Manual" if transmission is None else transmission
        brand = "Maruti" if not brand else brand


        # --------------------------------------------------
        # Create dataframe from user input
        # --------------------------------------------------
        new_car = pd.DataFrame({
            "year": [year],
            "km_driven": [km_driven],
            "owner": [owner],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "brand": [brand]
        })


        # --------------------------------------------------
        # One-hot encode categorical variables
        # --------------------------------------------------
        new_car_encoded = pd.get_dummies(
            new_car,
            columns=[
                "fuel",
                "seller_type",
                "transmission",
                "brand"
            ]
        )


        # --------------------------------------------------
        # Make sure the input has exactly the same
        # features as the trained model
        # --------------------------------------------------
        new_car_encoded = new_car_encoded.reindex(
            columns=model_features,
            fill_value=0
        )


        # --------------------------------------------------
        # Predict log selling price
        # --------------------------------------------------
        predicted_log_price = model.predict(new_car_encoded)


        # --------------------------------------------------
        # Convert log price back to original price
        # --------------------------------------------------
        predicted_price = np.exp(predicted_log_price[0])


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------
        return f"Predicted Selling Price: ₹{predicted_price:,.2f}"


    except Exception as e:

        return f"Prediction error: {str(e)}"


# --------------------------------------------------
# Run the application
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
