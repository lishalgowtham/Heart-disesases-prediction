import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =======================
# Load and preprocess data
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv('hearts.csv')
    le = LabelEncoder()
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()

# Feature and target split
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Model training
model = GaussianNB()
model.fit(X_train, y_train)

# =======================
# Streamlit UI
# =======================
st.title("💓 Heart Disease Prediction App")
st.write("Enter the patient's details below to predict the risk of heart disease.")

# Sidebar inputs
age = st.slider('Age', 18, 100, 29)
sex = st.selectbox('Sex', ['Male', 'Female'])  # 1 = Male, 0 = Female
chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])  # 0–3
resting_bp = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
cholesterol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])  # 0 or 1
resting_ecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T wave abnormality', 'Left Ventricular Hypertrophy'])  # 0–2
max_hr = st.slider('Maximum Heart Rate Achieved', 60, 210, 150)
exercise_angina = st.selectbox('Exercise-induced Angina', ['No', 'Yes'])  # 0 or 1
oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 6.0, 1.0, 0.1)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])  # 0–2

# Mapping categorical values to numbers (based on LabelEncoder)
sex_map = {'Male': 1, 'Female': 0}
cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
fbs_map = {'No': 0, 'Yes': 1}
ecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left Ventricular Hypertrophy': 2}
angina_map = {'No': 0, 'Yes': 1}
slope_map = {'Up': 2, 'Flat': 1, 'Down': 0}

# Create input for model
user_input = np.array([[
    age,
    sex_map[sex],
    cp_map[chest_pain],
    resting_bp,
    cholesterol,
    fbs_map[fasting_bs],
    ecg_map[resting_ecg],
    max_hr,
    angina_map[exercise_angina],
    oldpeak,
    slope_map[st_slope]
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(user_input)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease. Please consult a doctor.")
    else:
        st.success("✅ The patient is likely healthy. No signs of heart disease detected.")

# Show model accuracy
if st.checkbox("Show Model Accuracy"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
