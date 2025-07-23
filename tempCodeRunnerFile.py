import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import openai



# Set up the OpenAI API key
openai.api_key = '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'

# Function to generate response from OpenAI
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or any other model you prefer
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# loading the models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
liver_model = joblib.load('models/liver_model.sav')

# Sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Chatbot',
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Liver prediction',
    ],
    icons=['chat', 'activity', 'heart', 'person'],
    default_index=0)

# Chatbot page
if selected == 'Chatbot':
    st.title("ChatGPT with Streamlit")
    st.write("This is an example of integrating ChatGPT with a Streamlit app.")
    
    # Input text box for user prompt
    user_input = st.text_input("You:", "")
    
    if st.button("Send"):
        if user_input:
            with st.spinner("Generating response..."):
                response = generate_response(user_input)
                st.text_area("ChatGPT:", response, height=200)
        else:
            st.warning("Please enter some text.")

# Diabetes prediction page
if selected == 'Diabetes Prediction':
    st.title("Diabetes disease prediction")
    image = Image.open('d3.jpg')
    st.image(image, caption='diabetes disease prediction')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure value")
    with col1:
        SkinThickness = st.number_input("Skin thickness value")
    with col2:
        Insulin = st.number_input("Insulin value")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.number_input("Diabetes pedigree function value")
    with col2:
        Age = st.number_input("Age")
    
    # Prediction
    diabetes_dig = ''
    if st.button("Diabetes test result"):
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]]
        )
        if diabetes_prediction[0] == 1:
            diabetes_dig = "We are really sorry to say but it seems like you are Diabetic."
            image = Image.open('positive.jpg')
        else:
            diabetes_dig = 'Congratulations, You are not diabetic.'
            image = Image.open('negative.jpg')
        st.image(image, caption='')
        st.success(name + ', ' + diabetes_dig)

# Heart prediction page
if selected == 'Heart disease Prediction':
    st.title("Heart disease prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='Heart failure')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age")
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 0
    with col3:
        cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
        cp = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")
    with col2:
        chol = st.number_input("Serum Cholesterol")
    with col3:
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        restecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
    with col1:
        thalach = st.number_input("Max Heart Rate Achieved")
    with col2:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    with col3:
        slope = st.selectbox("Peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
        slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    with col1:
        ca = st.number_input("Number of major vessels (0–3) colored by fluoroscopy")
    with col2:
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
        thal = ["Normal", "Fixed defect", "Reversible defect"].index(thal)
    with col3:
        exang = st.checkbox('Exercise induced angina')
        exang = 1 if exang else 0
    with col1:
        fbs = st.checkbox('Fasting blood sugar > 120 mg/dl')
        fbs = 1 if fbs else 0
    
    # Prediction
    heart_dig = ''
    if st.button("Heart test result"):
        heart_prediction = heart_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        )
        if heart_prediction[0] == 1:
            heart_dig = 'We are really sorry to say but it seems like you have Heart Disease.'
            image = Image.open('positive.jpg')
        else:
            heart_dig = "Congratulations, You don't have Heart Disease."
            image = Image.open('negative.jpg')
        st.image(image, caption='')
        st.success(name + ', ' + heart_dig)

# Liver prediction page
if selected == 'Liver prediction':
    st.title("Liver disease prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        Sex = st.selectbox("Gender", ["Male", "Female"])
        Sex = 0 if Sex == "Male" else 1
    with col2:
        age = st.number_input("Enter your age")
    with col3:
        Total_Bilirubin = st.number_input("Enter your Total Bilirubin")
    with col1:
        Direct_Bilirubin = st.number_input("Enter your Direct Bilirubin")
    with col2:
        Alkaline_Phosphotase = st.number_input("Enter your Alkaline Phosphotase")
    with col3:
        Alamine_Aminotransferase = st.number_input("Enter your Alamine Aminotransferase")
    with col1:
        Aspartate_Aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")
    with col2:
        Total_Protiens = st.number_input("Enter your Total Proteins")
    with col3:
        Albumin = st.number_input("Enter your Albumin")
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Enter your Albumin and Globulin Ratio")
    
    # Prediction
    liver_dig = ''
    if st.button("Liver test result"):
        liver_prediction = liver_model.predict(
            [[Sex, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]]
        )
        if liver_prediction[0] == 1:
            liver_dig = "We are really sorry to say but it seems like you have liver disease."
            image = Image.open('positive.jpg')
        else:
            liver_dig = "Congratulations, You don't have liver disease."
            image = Image.open('negative.jpg')
        st.image(image, caption='')
        st.success(name + ', ' + liver_dig)
