import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load or train the model
try:
    model = joblib.load('car_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Loaded pre-trained model and scaler.")
except FileNotFoundError:
    print("Model files not found. Training new model...")
    # Load the dataset
    data = pd.read_csv('car.csv')  # Replace with your CSV file path

    # Feature Engineering
    data['Years_of_Service'] = 2025 - data['Year']
    data.drop(['Year', 'Car_Name'], axis=1, inplace=True)

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

    # Check for missing values
    if data.isnull().any().any():
        print("Missing values found. Imputing with mean for numerical columns...")
        numerical_cols = ['Present_Price', 'Kms_Driven', 'Years_of_Service', 'Owner']
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    # Split features and target
    X = data.drop('Selling_Price', axis=1)
    y = data['Selling_Price']

    # Feature scaling
    scaler = StandardScaler()
    numerical_cols = ['Present_Price', 'Kms_Driven', 'Years_of_Service']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Save the model and scaler
    joblib.dump(model, 'car_price_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Save expected columns for prediction
    expected_cols = X.columns
    joblib.dump(expected_cols, 'expected_cols.pkl')
else:
    expected_cols = joblib.load('expected_cols.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract and validate inputs
        def validate_float(value, field):
            try:
                if value is None or value.strip() == '':
                    raise ValueError(f"{field} cannot be empty.")
                return float(value)
            except ValueError:
                raise ValueError(f"Invalid {field}. Please enter a valid number.")

        def validate_int(value, field):
            try:
                if value is None or value.strip() == '':
                    raise ValueError(f"{field} cannot be empty.")
                return int(value)
            except ValueError:
                raise ValueError(f"Invalid {field}. Please enter a valid integer.")

        def validate_categorical(value, field, valid_options):
            value = value.capitalize() if value else ''
            if value not in valid_options:
                raise ValueError(f"Invalid {field}. Please choose from {valid_options}.")
            return value

        present_price = validate_float(data.get('present_price'), "Present Price")
        kms_driven = validate_float(data.get('kms_driven'), "Kilometers Driven")
        owner = validate_int(data.get('owner'), "Number of Previous Owners")
        years_of_service = validate_float(data.get('years_of_service'), "Years of Service")
        fuel_type = validate_categorical(data.get('fuel_type'), "Fuel Type", ['Petrol', 'Diesel', 'CNG'])
        seller_type = validate_categorical(data.get('seller_type'), "Seller Type", ['Dealer', 'Individual'])
        transmission = validate_categorical(data.get('transmission'), "Transmission", ['Manual', 'Automatic'])

        # Create input dataframe
        input_data = pd.DataFrame({
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Owner': [owner],
            'Years_of_Service': [years_of_service],
            'Fuel_Type_Diesel': [1 if fuel_type == 'Diesel' else 0],
            'Fuel_Type_Petrol': [1 if fuel_type == 'Petrol' else 0],
            'Seller_Type_Individual': [1 if seller_type == 'Individual' else 0],
            'Transmission_Manual': [1 if transmission == 'Manual' else 0]
        })

        # Ensure all expected columns are present
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match training data
        input_data = input_data[expected_cols]

        # Scale numerical features
        numerical_cols = ['Present_Price', 'Kms_Driven', 'Years_of_Service']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Predict
        predicted_price = model.predict(input_data)[0]
        return jsonify({'predicted_price': round(predicted_price, 2)})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)