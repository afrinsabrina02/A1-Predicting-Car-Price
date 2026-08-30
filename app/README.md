# Car Price Prediction

## Project Overview

This project focuses on predicting the selling price of used cars using machine learning techniques. The dataset contains information about different cars, including their manufacturing year, kilometers driven, fuel type, seller type, transmission, ownership history, mileage, engine capacity, maximum power, and number of seats.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model training, model evaluation, feature importance analysis, hyperparameter tuning, and deployment of the trained model through a web application.

## Dataset

The dataset used for this project is `Cars.csv`.

The original dataset contains 8,128 records and 13 columns. During preprocessing, missing values and unsuitable records were handled, categorical features were encoded, and relevant numerical features were extracted.

After preprocessing, the final dataset contains:

- 7,814 records
- 12 features

The target variable is:

- `selling_price`

## Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary information from the dataset.
- Converted `mileage`, `engine`, and `max_power` into numerical values.
- Removed the `torque` feature.
- Extracted the car brand from the `name` column.
- Converted owner categories into numerical values.
- Removed unsuitable records such as Test Drive Cars.
- Handled missing values.
- Applied log transformation to the target variable.
- Applied one-hot encoding to categorical variables.
- Split the data into training and testing sets.

The final feature matrix contains 41 features.

## Machine Learning Models

Three approaches were evaluated:

1. Baseline model
2. Decision Tree Regressor
3. Random Forest Regressor

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Model Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Baseline | 113,254.01 | 58,090,971,378.03 | 241,020.69 | 0.9142 |
| Decision Tree | 79,405.40 | 22,217,566,518.09 | 149,055.58 | 0.9672 |
| Random Forest | 72,867.36 | 18,894,189,748.36 | 137,456.14 | 0.9721 |
| Tuned Random Forest | 63,763.02 | 16,175,503,336.27 | 127,182.95 | 0.9761 |

The tuned Random Forest achieved the best performance among the tested models.

## Hyperparameter Tuning

Hyperparameter tuning was performed for the Random Forest model.

The best parameters were:

- `max_depth = 15`
- `min_samples_leaf = 1`
- `n_estimators = 200`

The best cross-validation R² score was approximately:

`0.9363`

After tuning, the model achieved a test R² score of approximately:

`0.9761`

## Feature Importance

The most important features identified by the Random Forest model were:

- `max_power`
- `year`
- `engine`
- `km_driven`
- `mileage`

Among these, `max_power` and `year` had the highest feature importance.

The feature importance analysis indicates that the car's maximum power and manufacturing year played particularly important roles in predicting its selling price.

## Model Inference

The trained model was saved as:

`car_price_model.pkl`

The feature names used during training were saved as:

`model_features.pkl`

These files are used by the deployment application to load the trained model and generate predictions for new car information.

## Web Application

A Dash web application was developed for Task 3.

The application allows a user to enter car information such as:

- Year
- Kilometers driven
- Owner
- Mileage
- Engine
- Maximum power
- Number of seats
- Fuel type
- Seller type
- Transmission
- Brand

After entering the information and clicking **Predict Selling Price**, the application displays the predicted selling price.

## Docker Deployment

The application was containerized using Docker.

The deployment files are located inside the `app` directory:

```text
app/
├── .Dockerfile
├── docker-compose.yaml
└── code/
    └── app.py