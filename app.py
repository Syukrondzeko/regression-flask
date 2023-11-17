from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load both the model and the transformer
model = pickle.load(open('decision_tree_regressor.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("health_bill.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Collect input features from the form, ensuring all inputs are present
    age = request.form.get('Age', type=int)
    if age is None or age < 1:
        return render_template('health_bill.html', pred='Please enter a valid age.')

    gender = request.form.get('Gender')
    blood_type = request.form.get('BloodType')
    medical_condition = request.form.get('MedicalCondition')
    
    # Create a DataFrame from the input features to match the expected input for the transformer
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Blood Type': blood_type,
        'Medical Condition': medical_condition
    }])
    
    # Ensure no None values - handle missing inputs as needed
    if input_df.isnull().values.any():
        return render_template('health_bill.html', pred='Please fill in all required fields.')
    
    # Preprocess the inputs using the transformer
    transformed_features = transformer.transform(input_df)
    
    # Predict using the preprocessed features
    prediction = model.predict(transformed_features)
    output = round(prediction[0], 2)
    
    return render_template('health_bill.html', pred=f'Predicted Billing Amount is {output}')

if __name__ == '__main__':
    app.run(debug=True)
