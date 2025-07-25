import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 75, 30)
workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp", "Gov", "Others"])
fnlwgt = st.sidebar.number_input("FNLWGT", min_value=10000, max_value=1000000, value=150000)
educational_num = st.sidebar.slider("Education Number", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced", "Separated", "Widowed"])
occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Sales", "Exec-managerial", "Other-service", "Craft-repair", "Others"])
relationship = st.sidebar.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"])
race = st.sidebar.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=100000, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
native_country = st.sidebar.selectbox("Country", ["United-States", "India", "Mexico", "Philippines", "Other"])

input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Label encoding mapping (used same as during training)
label_maps = {
    'workclass': {"Private": 0, "Self-emp": 1, "Gov": 2, "Others": 3},
    'marital-status': {"Married": 0, "Single": 1, "Divorced": 2, "Separated": 3, "Widowed": 4},
    'occupation': {"Tech-support": 0, "Sales": 1, "Exec-managerial": 2, "Other-service": 3, "Craft-repair": 4, "Others": 5},
    'relationship': {"Husband": 0, "Not-in-family": 1, "Own-child": 2, "Unmarried": 3, "Wife": 4, "Other-relative": 5},
    'race': {"White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4},
    'gender': {"Male": 0, "Female": 1},
    'native-country': {"United-States": 0, "India": 1, "Mexico": 2, "Philippines": 3, "Other": 4}
}

for col, mapping in label_maps.items():
    input_df[col] = input_df[col].map(mapping)

st.write("### 🔎 Input Data")
st.write(input_df)

if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"🚫 Prediction Failed: {e}")

# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    try:
        for col, mapping in label_maps.items():
            if col in batch_data.columns:
                batch_data[col] = batch_data[col].map(mapping)

        batch_data = batch_data[input_df.columns]
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"🚫 Batch prediction failed: {e}")
