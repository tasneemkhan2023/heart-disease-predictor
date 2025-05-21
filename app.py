import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('heart_disease_model.pkl')

# Feature descriptions

with st.expander("🧠 **Feature Guide: Understand Your Inputs**"):
    st.markdown("""
### 🧠 Feature Guide: Understand Your Inputs

Below is a guide to help you understand each input, how to measure it, and how it contributes to predicting heart disease risk.

---

#### 🔢 **Age**
- **What**: Your age in years.
- **How to get it**: You already know this 🙂 (or you can check your birth certificate)
- **Why it matters**: Heart disease risk increases with age, especially after 40.

---

#### 🚻 **Sex**
- **Options**: Female, Male
- **Why it matters**: Males are generally at higher risk of early heart disease, though risk increases for women post-menopause.

---

#### ⚖️ **BMI (Body Mass Index)**
- **What**: Weight-to-height ratio: `BMI = weight (kg) / height² (m²)`
- **How to get it**: Use an online [BMI calculator](https://www.calculator.net/bmi-calculator.html) or weigh yourself and measure your height.
- **Why it matters**: Higher BMI indicates overweight or obesity, increasing heart risk.

---

#### 🍬 **Glucose Level**
- **What**: Blood sugar level (in mg/dL)
- **How to get it**: Check with a glucometer or from your blood test report.
- **Why it matters**: High glucose is a sign of diabetes, which is a major heart risk factor.

---

#### 🧠 **History of Stroke**
- **Options**: Yes, No
- **Why it matters**: Previous stroke is a strong indicator of vascular issues and heart disease risk.

---

#### 💢 **Hypertension**
- **What**: Do you have high blood pressure?
- **How to know**: If a doctor has diagnosed you or your systolic BP is consistently ≥140 mmHg.
- **Why it matters**: High blood pressure damages artery walls and increases heart risk.

---

#### 🩸 **Total Cholesterol**
- **Unit**: mg/dL  
- **Normal Range**: Under 200 mg/dL
- **How to get it**: From your lipid profile/blood test report.
- **Why it matters**: High cholesterol can lead to plaque buildup in arteries (atherosclerosis).

---

#### ❤️ **Heart Rate**
- **Unit**: beats per minute (bpm)
- **How to measure**: Use a fitness band, smart watch, or place two fingers on your wrist/neck and count beats for 60 seconds.
- **Normal Range**: 60–100 bpm
- **Why it matters**: Abnormal resting heart rate may indicate cardiovascular problems.

---

#### 🩺 **Systolic BP (Top Number)**
- **What**: Pressure during heartbeats
- **Normal**: ~120 mmHg
- **How to measure**: With a digital blood pressure monitor.
- **Why it matters**: High systolic pressure increases strain on arteries.

---

#### 💨 **Diastolic BP (Bottom Number)**
- **What**: Pressure between heartbeats
- **Normal**: ~80 mmHg
- **Why it matters**: Elevated values may point to chronic heart stress.

---

#### 💊 **BP Medication**
- **What**: Are you taking medications for blood pressure?
- **Why it matters**: Indicates pre-existing hypertension being managed—still a risk factor.

---

#### 🚬 **Current Smoker**
- **Options**: Yes , No 
- **Why it matters**: Smoking causes narrowing of blood vessels and increases plaque buildup.

---

#### 🚬 **Cigarettes Per Day**
- **Range**: 0–40 (use slider or type in)
- **How to estimate**: Average number of cigarettes smoked daily.
- **Why it matters**: The more you smoke, the greater the cardiovascular damage.

---

#### 🍩 **Diabetes**
- **Checkbox**: Tick if you have been diagnosed.
- **Why it matters**: Diabetes significantly raises the risk of heart-related complications.

---

### 💡 Pro Tip:
To get the most accurate prediction:
- Use recent medical data (within 6 months).
- Consult your doctor if unsure about any metric.
- Avoid guessing or leaving fields blank if possible.
""")


#User input form
with st.form("patient_form"):
    # Existing features
    age = st.slider("Age", 30, 80, 45)
    male = st.radio("Sex", ["Female", "Male"])
    BMI = st.number_input("BMI", 15.0, 40.0, 25.0)
    glucose = st.number_input("Glucose Level", 60, 300, 100)
    prevalentStroke = st.radio("History of Stroke", ["No", "Yes"])
    prevalentHyp = st.radio("Hypertension", ["No", "Yes"])
    totChol = st.number_input("Total Cholesterol", 100, 600, 200)
    heartRate = st.slider("Heart Rate", 50, 120, 75)
    sysBP = st.number_input("Systolic BP", 80, 200, 120)
    diaBP = st.number_input("Diastolic BP", 60, 120, 80)
    BPMeds = st.radio("BP Medication", ["No", "Yes"])
    currentSmoker = st.radio("Current Smoker", ["No", "Yes"])
    cigsPerDay = st.slider("Cigarettes/Day", 0, 40, 0)
    diabetes = st.checkbox("Diabetes")

    submitted = st.form_submit_button("Predict")

# Convert to DataFrame
if submitted:
    input_data = pd.DataFrame([[
        1 if male == "Male" else 0,
        age,
        1 if currentSmoker == "Yes" else 0,
        cigsPerDay,
        1 if BPMeds == "Yes" else 0,
        1 if prevalentStroke == "Yes" else 0,
        1 if prevalentHyp == "Yes" else 0,
        1 if diabetes else 0,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]], columns=[
        'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
        'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ])
    # Predict
    proba = model.predict_proba(input_data)[0][1]
    risk = "High Risk" if proba > 0.3 else "Low Risk"
    
    # Display
    st.metric("10-Year CHD Probability", f"{proba*100:.1f}%", risk)
    st.progress(proba)

    print("Training features:", model.feature_names_in_)