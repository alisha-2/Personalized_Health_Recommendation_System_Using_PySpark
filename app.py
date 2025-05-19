import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import requests

# Cache SparkSession to improve performance
@st.cache_resource
def init_spark():
    spark = SparkSession.builder.appName("HealthRecommendationApp").getOrCreate()
    return spark

spark = init_spark()

# Load datasets
df = spark.read.option("header", "true").csv("datasets/final.csv", inferSchema=True)
try:
    symptoms_df = pd.read_csv("datasets/symptoms.csv", quoting=csv.QUOTE_ALL)
except pd.errors.ParserError as e:
    st.error(f"Error loading symptoms.csv: {e}")
    st.stop()
model = ALSModel.load("models/als_model")

# Create a mapping of symptoms to syd values from symptoms.csv
symptom_to_syd = {str(row["symptom"]).strip().lower(): int(row["syd"]) for _, row in symptoms_df.iterrows()}

# Extract all unique diagnoses from final.csv
unique_diagnoses = df.select("diagnose").distinct().toPandas()["diagnose"].tolist()

# Function to fetch health recommendations dynamically from a medical API
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_health_recommendation(diagnosis):
    """
    Fetch health recommendations for a given diagnosis from a medical API.
    Returns a dictionary with recommendations for different severity levels and demographics.
    """
    # Simulate API call (replace with a real API endpoint)
    try:
        # Placeholder for a real API call
        # Example: response = requests.get(f"https://api.medicaldatabase.example/recommendations?diagnosis={diagnosis}", timeout=10)
        # response.raise_for_status()
        # api_response = response.json()

        # Mock API response for demonstration
        mock_api_response = {
            "Diabetes": {
                "mild": "Monitor your blood sugar levels daily, follow a balanced diet with low sugar intake, and engage in light exercise like walking for 30 minutes a day. Learn more at: https://www.cdc.gov/diabetes/managing/index.html",
                "severe": "Consult a healthcare provider immediately for personalized insulin management. Follow a strict low-carb diet, avoid sugary drinks, and exercise regularly. Learn more at: https://www.who.int/health-topics/diabetes",
                "elderly": "Consult a healthcare provider for tailored management, as older adults may need adjusted insulin doses. Focus on a low-carb diet and avoid overexertion during exercise. Learn more at: https://www.who.int/health-topics/diabetes",
                "child": "Consult a pediatric endocrinologist for a child-friendly diabetes management plan. Ensure a balanced diet and monitor blood sugar closely. Learn more at: https://www.cdc.gov/diabetes/managing/index.html"
            },
            "Asthma": {
                "mild": "Avoid allergens like dust and pollen, use your inhaler as prescribed, and practice breathing exercises. Learn more at: https://www.cdc.gov/asthma/managing.html",
                "severe": "Seek medical attention if symptoms worsen. Keep your rescue inhaler handy, avoid triggers, and consider a humidifier to ease breathing. Learn more at: https://www.who.int/news-room/fact-sheets/detail/asthma",
                "elderly": "Seek medical attention if breathing difficulties persist, as older adults may have reduced lung capacity. Use a humidifier and avoid cold air. Learn more at: https://www.who.int/news-room/fact-sheets/detail/asthma",
                "child": "Consult a pediatrician for an asthma action plan tailored for children. Ensure they use their inhaler correctly and avoid allergens. Learn more at: https://www.cdc.gov/asthma/managing.html"
            },
            "Gastric ulcer/stomach ulcer": {
                "mild": "Avoid spicy foods, alcohol, and NSAIDs like ibuprofen. Eat smaller, frequent meals and consider over-the-counter antacids. Learn more at: https://www.mayoclinic.org/diseases-conditions/stomach-ulcer/diagnosis-treatment/drc-20373171",
                "severe": "Consult a doctor immediately, especially if you experience severe pain, vomiting blood, or black stools. You may need medication to reduce stomach acid or treat H. pylori infection. Learn more at: https://www.mayoclinic.org/diseases-conditions/stomach-ulcer/diagnosis-treatment/drc-20373171",
                "elderly": "Consult a doctor immediately, as older adults are at higher risk for complications. Strictly avoid NSAIDs and alcohol, and use antacids under medical guidance. Learn more at: https://www.mayoclinic.org/diseases-conditions/stomach-ulcer/diagnosis-treatment/drc-20373171",
                "child": "Consult a pediatrician for child-appropriate treatment. Avoid spicy foods and ensure smaller meals. Learn more at: https://www.mayoclinic.org/diseases-conditions/stomach-ulcer/diagnosis-treatment/drc-20373171",
                "medical_history": "If you have a history of H. pylori infection, consult a doctor for targeted treatment, as this may be a recurring issue. Avoid NSAIDs and follow a bland diet. Learn more at: https://www.mayoclinic.org/diseases-conditions/stomach-ulcer/diagnosis-treatment/drc-20373171"
            }
        }

        # Simulate fetching the recommendation for the given diagnosis
        diagnosis_normalized = str(diagnosis).strip().lower()
        for key in mock_api_response:
            if key.lower() == diagnosis_normalized:
                return mock_api_response[key]

        # Fallback if the diagnosis is not found in the API response
        return {
            "mild": f"Monitor your symptoms for {diagnosis} closely, stay hydrated, and rest. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "severe": f"Seek medical attention for {diagnosis} promptly to address your symptoms. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "elderly": f"Seek medical attention for {diagnosis}, as older adults may need specialized care. Stay hydrated and rest. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "child": f"Consult a pediatrician for {diagnosis} to ensure child-appropriate care. Monitor symptoms closely. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "medical_history": f"Seek medical attention for {diagnosis}, especially if you have a relevant medical history that may complicate symptoms. For more information, visit: https://www.mayoclinic.org/diseases-conditions"
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch recommendations for {diagnosis}: {e}")
        # Fallback recommendation if API fails
        return {
            "mild": f"Monitor your symptoms for {diagnosis} closely, stay hydrated, and rest. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "severe": f"Seek medical attention for {diagnosis} promptly to address your symptoms. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "elderly": f"Seek medical attention for {diagnosis}, as older adults may need specialized care. Stay hydrated and rest. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "child": f"Consult a pediatrician for {diagnosis} to ensure child-appropriate care. Monitor symptoms closely. For more information, visit: https://www.mayoclinic.org/diseases-conditions",
            "medical_history": f"Seek medical attention for {diagnosis}, especially if you have a relevant medical history that may complicate symptoms. For more information, visit: https://www.mayoclinic.org/diseases-conditions"
        }

# Streamlit app
st.title("🩺 Personalized Health Recommendation System")
st.write("Select your symptoms to get a diagnosis and health recommendations. " \
"Note: This tool is for informational purposes only and not a substitute for professional medical advice.")

# Display last updated timestamp
if os.path.exists("datasets/last_updated.txt"):
    with open("datasets/last_updated.txt", "r") as f:
        last_updated = f.read()
    st.write(f"Dataset last updated on: {last_updated}")
else:
    st.write("Dataset update status: Not available")

# Demographic inputs
st.write("### Provide Your Demographic Information")
age = st.number_input("Age:", min_value=0, max_value=120, value=30, step=1)
gender = st.selectbox("Gender:", options=["Male", "Female", "Other"])
medical_history = st.text_input("Medical History (e.g., H. pylori infection, asthma):", "")

# Symptom selection with severity
symptom_list = [str(sym).strip() for sym in symptoms_df["symptom"].dropna().unique().tolist()]
selected_symptoms = st.multiselect("Select symptoms:", symptom_list)

# Add severity sliders for each selected symptom
severity_dict = {}
if selected_symptoms:
    st.write("Rate the severity of each symptom (1 = mild, 5 = severe):")
    for symptom in selected_symptoms:
        severity = st.slider(f"Severity of {symptom}", 1, 5, 3, key=symptom)
        severity_dict[symptom] = severity

# Diagnosis prediction
if st.button("Get Diagnosis"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        # Map selected symptoms to syd values
        syds = []
        for sym in selected_symptoms:
            sym_normalized = str(sym).strip().lower()
            if sym_normalized in symptom_to_syd:
                syds.append(symptom_to_syd[sym_normalized])
            else:
                st.warning(f"Symptom '{sym}' not found in mapping. Skipping.")

        syds = [s for s in syds if s is not None]

        if not syds:
            st.error("No matching symptoms found in the dataset.")
        else:
            # Filter final.csv using the syd values
            input_df = df.filter(df["syd"].isin(syds)).select("syd", "did", "diagnose", "wei")
            predictions = model.transform(input_df).select("did", "diagnose", "prediction").distinct()
            results = predictions.orderBy("prediction", ascending=False).limit(5).toPandas()

            if results.empty:
                st.error("No predictions available for the selected symptoms.")
            else:
                results["Rank"] = range(len(results))
                st.write("### Top 5 Predicted Diagnoses")
                st.table(results[["Rank", "diagnose", "prediction"]])

                # Bar chart visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(results["diagnose"], results["prediction"], color="skyblue")
                ax.invert_yaxis()
                ax.set_xlabel("Prediction Score")
                ax.set_title("Top 5 Predicted Diagnoses")
                st.pyplot(fig)

                # Health recommendation with demographic consideration
                top_diagnosis = results.iloc[0]["diagnose"]
                top_diagnosis_normalized = str(top_diagnosis).strip().lower()

                # Fetch recommendation dynamically
                health_tips = fetch_health_recommendation(top_diagnosis)

                avg_severity = sum(severity_dict.values()) / len(severity_dict) if severity_dict else 3
                severity_level = "severe" if avg_severity >= 3 else "mild"

                # Determine recommendation key based on demographics
                recommendation_key = severity_level
                if age >= 65:
                    recommendation_key = "elderly"
                elif age < 18:
                    recommendation_key = "child"

                # Further adjust recommendation if medical history is relevant
                if medical_history and "medical_history" in health_tips:
                    medical_history_lower = medical_history.lower()
                    if "gastric ulcer" in top_diagnosis_normalized and "h. pylori" in medical_history_lower:
                        recommendation_key = "medical_history"

                st.write("### Health Recommendation")
                st.write(f"**Diagnosis**: {top_diagnosis}")
                st.write(f"**Recommendation (based on {recommendation_key} considerations)**: {health_tips[recommendation_key]}")