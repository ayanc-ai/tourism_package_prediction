!pip install streamlit -q

import streamlit as st
import pandas as pd
import joblib
import os

# Define the path to the trained model
MODEL_PATH = 'tourism_project/deployment/model.joblib'

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
else:
    # Load the trained model
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        model = None

# Streamlit App Title
st.title('Tourism Package Purchase Prediction 🌍✈️')
st.write('Enter customer details to predict if they will purchase the Wellness Tourism Package.')

if model is not None:
    # Define input widgets for each feature
    st.header('Customer Information')

    # Numerical features
    age = st.slider('Age', 18, 80, 30)
    city_tier = st.selectbox('City Tier', [1, 2, 3])
    duration_of_pitch = st.slider('Duration of Pitch (minutes)', 0.0, 60.0, 10.0)
    number_of_person_visiting = st.slider('Number of Persons Visiting', 1, 10, 1)
    preferred_property_star = st.slider('Preferred Property Star Rating', 1, 5, 3)
    number_of_trips = st.slider('NumberOfTrips', 0, 20, 2)
    passport = st.radio('Has Passport?', ['No', 'Yes'])
    own_car = st.radio('Owns a Car?', ['No', 'Yes'])
    number_of_children_visiting = st.slider('Number of Children Visiting', 0, 5, 0)
    monthly_income = st.number_input('Monthly Income (USD)', min_value=0.0, value=25000.0, step=1000.0)
    pitch_satisfaction_score = st.slider('Pitch Satisfaction Score (1-5)', 1, 5, 3)
    number_of_followups = st.slider('Number of Follow-ups', 0, 10, 2)

    # Categorical features
    typeof_contact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
    occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Government', 'Student'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
    designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director', 'Junior Executive'])
    product_pitched = st.selectbox('Product Pitched', ['Package A', 'Package B', 'Package C', 'Package D', 'Package E'])

    # Map 'Yes'/'No' to 1/0 for binary features
    passport_val = 1 if passport == 'Yes' else 0
    own_car_val = 1 if own_car == 'Yes' else 0

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [typeof_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'PreferredPropertyStar': [preferred_property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [passport_val],
        'OwnCar': [own_car_val],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'ProductPitched': [product_pitched],
        'NumberOfFollowups': [number_of_followups]
    })

    # Reorder columns to match the training data's feature order if necessary
    # This is handled by the ColumnTransformer in the pipeline, but it's good practice
    # to ensure consistency if raw feature names matter for specific preprocessing steps.
    # For pipelines, as long as names are correct, order often doesn't strictly matter for ColumnTransformer.

    if st.button('Predict Purchase'):
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            st.subheader('Prediction Result:')
            if prediction[0] == 1:
                st.success('The customer is likely to purchase the Wellness Tourism Package! 🎉')
                st.markdown(f"**Probability of Purchase: {prediction_proba[0][1]*100:.2f}%**")
            else:
                st.info('The customer is not likely to purchase the Wellness Tourism Package. 😔')
                st.markdown(f"**Probability of Purchase: {prediction_proba[0][1]*100:.2f}%**")

            # Optional: Display input data for debugging/verification
            # st.write('Input Data:')
            # st.dataframe(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please ensure all inputs are valid and the model is correctly loaded.")
