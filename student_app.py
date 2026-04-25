import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# 📂 PATHS (YOUR ORIGINAL PATHS)
# -----------------------------
model_path = r"C:\Users\suraj\Downloads\model\logistic_model.pkl"
scaler_path = r"C:\Users\suraj\Downloads\model\scaler.pkl"
columns_path = r"C:\Users\suraj\Downloads\model\columns.pkl"
data_path = r"C:\Users\suraj\Downloads\Student Depression Dataset.csv"

# -----------------------------
# 🧪 DEBUG CHECK
# -----------------------------
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"❌ Scaler file not found: {scaler_path}")
    st.stop()

if not os.path.exists(columns_path):
    st.error(f"❌ Columns file not found: {columns_path}")
    st.stop()

if not os.path.exists(data_path):
    st.error(f"❌ Dataset file not found: {data_path}")
    st.stop()

# -----------------------------
# 📦 LOAD MODEL, SCALER, COLUMNS & DATA
# -----------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
columns = joblib.load(columns_path)

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# -----------------------------
# 🎛️ SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation")
option = st.sidebar.radio("Go to", ["Dataset", "EDA", "Prediction"])

# -----------------------------
# 📊 DATASET SECTION
# -----------------------------
if option == "Dataset":
    st.title("📊 Dataset Overview")
    st.dataframe(df.head())
    st.write(df.describe())

# -----------------------------
# 📈 EDA SECTION
# -----------------------------
elif option == "EDA":
    st.title("📈 Exploratory Data Analysis")

    # Convert Depression if needed
    if df["Depression"].dtype == "object":
        df["Depression"] = df["Depression"].map({"Yes": 1, "No": 0})

    # Create Total Pressure
    df["Total Pressure"] = df["Academic Pressure"] + df["Work Pressure"]

    # Countplot
    st.subheader("Depression Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Depression", data=df, ax=ax1)
    ax1.set_xticklabels(["No", "Yes"])
    st.pyplot(fig1)

    # Boxplot
    st.subheader("Total Pressure vs Depression")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Depression", y="Total Pressure", data=df, ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# 🤖 PREDICTION SECTION
# -----------------------------
elif option == "Prediction":
    st.title("🤖 Depression Prediction")

    # Numerical Inputs (MATCH TRAINING)
    age = st.number_input("Age", 15, 40)
    academic_pressure = st.slider("Academic Pressure", 0, 10)
    work_pressure = st.slider("Work Pressure", 0, 10)
    cgpa = st.number_input("CGPA", 0.0, 10.0)

    study_satisfaction = st.slider("Study Satisfaction", 0, 10)
    job_satisfaction = st.slider("Job Satisfaction", 0, 10)

    sleep_duration = st.number_input("Sleep Duration (numeric)", 0.0, 12.0)
    work_hours = st.number_input("Work/Study Hours", 0.0, 24.0)

    # -----------------------------
    # 🔮 PREDICT
    # -----------------------------
    if st.button("Predict"):
        try:
            # Step 1: Create input
            input_df = pd.DataFrame({
                "Age": [age],
                "Academic Pressure": [academic_pressure],
                "Work Pressure": [work_pressure],
                "CGPA": [cgpa],
                "Study Satisfaction": [study_satisfaction],
                "Job Satisfaction": [job_satisfaction],
                "Sleep Duration": [sleep_duration],
                "Work/Study Hours": [work_hours],
                "Total Pressure": [academic_pressure + work_pressure]
            })

            # Step 2: Scale
            input_scaled = scaler.transform(input_df)

            # Step 3: Convert to DataFrame
            input_df_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)

            # Step 4: Align with training columns (🔥 CRITICAL)
            input_df_scaled = input_df_scaled.reindex(columns=columns, fill_value=0)

            # Debug info (optional)
            st.write(f"Model expects {len(columns)} features")
            st.write(f"Input given {input_df_scaled.shape[1]} features")

            # Step 5: Predict
            prediction = model.predict(input_df_scaled)[0]

            # Probability (optional)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df_scaled)[0][1]
                st.write(f"📊 Confidence: {prob*100:.2f}%")

            # Output
            if prediction == 1:
                st.error("⚠️ Possible Depression")
            else:
                st.success("😊 No Depression")

            st.info("This is not a medical diagnosis.")

        except Exception as e:
            st.error(f"❌ Error: {e}")