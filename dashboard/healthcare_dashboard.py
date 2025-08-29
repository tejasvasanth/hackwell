import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.supabase_client import supabase_client

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration (fallback for some endpoints)
API_BASE_URL = "http://localhost:8000"

class DataClient:
    """Client for interacting with Supabase and API backend"""
    
    def __init__(self):
        self.supabase = supabase_client
        self.session = requests.Session()
    
    def get_health_status(self) -> Dict:
        """Check Supabase connection status"""
        try:
            # Test Supabase connection
            result = self.supabase.client.table("patients").select("id").limit(1).execute()
            return {"status": "healthy", "data": {"message": "Supabase connected"}}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patients(self) -> Dict:
        """Get list of all patients from Supabase"""
        try:
            result = self.supabase.client.table("patients").select(
                "id, name, age, gender, smoking, exercise, bp, cholesterol, diabetes_status, created_at"
            ).execute()
            
            if result.data:
                return {"status": "success", "data": {"patients": result.data}}
            else:
                return {"status": "success", "data": {"patients": []}}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patient_data(self, patient_id: str) -> Dict:
        """Get detailed patient data from Supabase"""
        try:
            # Get patient basic info
            patient_result = self.supabase.client.table("patients").select("*").eq("id", patient_id).execute()
            
            if not patient_result.data:
                return {"status": "error", "message": "Patient not found"}
            
            patient = patient_result.data[0]
            
            # Get lab results
            lab_result = self.supabase.client.table("lab_results").select("*").eq("patient_id", patient_id).order("test_date", desc=True).limit(10).execute()
            
            # Get lifestyle data
            lifestyle_result = self.supabase.client.table("lifestyle_data").select("*").eq("patient_id", patient_id).order("recorded_date", desc=True).limit(10).execute()
            
            # Format data for dashboard compatibility
            formatted_data = {
                "id": patient["id"],
                "name": patient["name"],
                "age": patient["age"],
                "gender": patient["gender"],
                "demographics": {
                    "height": patient.get("height", 170),
                    "weight": patient.get("weight", 70),
                    "bmi": patient.get("bmi", 24.2),
                    "blood_type": patient.get("blood_type", "O+"),
                    "phone": patient.get("phone", "N/A"),
                    "emergency_contact": patient.get("emergency_contact", "N/A")
                },
                "lab_results": self._format_lab_results(lab_result.data),
                "lifestyle_data": self._format_lifestyle_data(lifestyle_result.data, patient)
            }
            
            return {"status": "success", "data": formatted_data}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _format_lab_results(self, lab_data: List[Dict]) -> Dict:
        """Format lab results for dashboard display"""
        if not lab_data:
            return {
                "cholesterol": 200, "hdl": 50, "ldl": 120, "triglycerides": 150,
                "glucose": 90, "hba1c": 5.5, "systolic_bp": 120, "diastolic_bp": 80,
                "last_test_date": "2024-01-01"
            }
        
        # Group by test type and get latest values
        latest_results = {}
        for result in lab_data:
            test_type = result["test_type"]
            if test_type not in latest_results or result["test_date"] > latest_results[test_type]["test_date"]:
                latest_results[test_type] = result
        
        return {
            "cholesterol": latest_results.get("cholesterol_total", {}).get("value", 200),
            "hdl": latest_results.get("cholesterol_hdl", {}).get("value", 50),
            "ldl": latest_results.get("cholesterol_ldl", {}).get("value", 120),
            "triglycerides": latest_results.get("triglycerides", {}).get("value", 150),
            "glucose": latest_results.get("glucose", {}).get("value", 90),
            "hba1c": latest_results.get("hba1c", {}).get("value", 5.5),
            "systolic_bp": latest_results.get("bp_systolic", {}).get("value", 120),
            "diastolic_bp": latest_results.get("bp_diastolic", {}).get("value", 80),
            "last_test_date": max([r["test_date"] for r in lab_data]) if lab_data else "2024-01-01"
        }
    
    def _format_lifestyle_data(self, lifestyle_data: List[Dict], patient: Dict) -> Dict:
        """Format lifestyle data for dashboard display"""
        if lifestyle_data:
            latest = lifestyle_data[0]
            return {
                "smoking_status": patient.get("smoking", "Never"),
                "alcohol_consumption": latest.get("alcohol_consumption", "None"),
                "exercise_frequency": patient.get("exercise", 3),
                "diet_type": latest.get("diet_type", "Balanced"),
                "sleep_hours": latest.get("sleep_hours", 7),
                "stress_level": latest.get("stress_level", 3),
                "family_history": patient.get("family_history", False)
            }
        else:
            return {
                "smoking_status": patient.get("smoking", "Never"),
                "alcohol_consumption": "None",
                "exercise_frequency": patient.get("exercise", 3),
                "diet_type": "Balanced",
                "sleep_hours": 7,
                "stress_level": 3,
                "family_history": patient.get("family_history", False)
            }
    
    def predict_patient_risk(self, patient_id: str) -> Dict:
        """Get risk prediction for specific patient (fallback to API)"""
        try:
            response = self.session.get(f"{API_BASE_URL}/predict/{patient_id}", timeout=10)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patient_recommendations(self, patient_id: str) -> Dict:
        """Get recommendations for specific patient (fallback to API)"""
        try:
            response = self.session.get(f"{API_BASE_URL}/recommendations/{patient_id}", timeout=10)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_monitoring_status(self) -> Dict:
        """Get monitoring system status (fallback to API)"""
        try:
            response = self.session.get(f"{API_BASE_URL}/monitoring/status", timeout=5)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Initialize data client
api_client = DataClient()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'selected_patient_id' not in st.session_state:
    st.session_state.selected_patient_id = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'patients_list' not in st.session_state:
    st.session_state.patients_list = []
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

def render_login_screen():
    """Render the patient selection/login screen"""
    st.markdown('<h1 class="main-header">🏥 Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown("### 👤 Patient Selection")
        st.markdown("Please select a patient to view their healthcare analytics.")
        
        # Fetch patients list
        with st.spinner("Loading patients..."):
            patients_result = api_client.get_patients()
        
        if patients_result["status"] == "success":
            patients_data = patients_result["data"]
            st.session_state.patients_list = patients_data["patients"]
            
            # Create patient selection dropdown
            patient_options = {}
            for patient in st.session_state.patients_list:
                display_name = f"{patient['name']} (ID: {patient['id']}) - {patient['age']}y, {patient['gender']}"
                patient_options[display_name] = patient['id']
            
            if patient_options:
                selected_display = st.selectbox(
                    "Select Patient:",
                    options=list(patient_options.keys()),
                    help="Choose a patient to view their healthcare data"
                )
                
                selected_patient_id = patient_options[selected_display]
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔐 Access Patient Dashboard", type="primary", use_container_width=True):
                        # Fetch patient data
                        with st.spinner("Loading patient data..."):
                            patient_result = api_client.get_patient_data(selected_patient_id)
                        
                        if patient_result["status"] == "success":
                            st.session_state.selected_patient_id = selected_patient_id
                            st.session_state.patient_data = patient_result["data"]
                            st.session_state.logged_in = True
                            st.success(f"✅ Successfully logged in as {st.session_state.patient_data['name']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load patient data: {patient_result['message']}")
            else:
                st.warning("No patients available in the system.")
        else:
            st.error(f"❌ Failed to load patients: {patients_result['message']}")
            st.info("Please ensure the API server is running and accessible.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with patient info and controls"""
    with st.sidebar:
        # Patient info
        if st.session_state.patient_data:
            st.markdown("### 👤 Current Patient")
            patient = st.session_state.patient_data
            st.markdown(f"**Name:** {patient['name']}")
            st.markdown(f"**ID:** {patient['id']}")
            st.markdown(f"**Age:** {patient['age']} years")
            st.markdown(f"**Gender:** {patient['gender']}")
            
            # Logout button
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                # Clear session state
                st.session_state.logged_in = False
                st.session_state.selected_patient_id = None
                st.session_state.patient_data = None
                st.session_state.prediction_results = None
                st.session_state.recommendations = None
                st.rerun()
        
        st.divider()
        
        # System Status
        st.markdown("### 🔧 System Status")
        
        # API Health Check
        health_status = api_client.get_health_status()
        if health_status["status"] == "healthy":
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Error: {health_status.get('message', 'Unknown error')}")
        
        # Monitoring Status
        monitoring_status = api_client.get_monitoring_status()
        if monitoring_status["status"] == "success":
            st.info("📊 Monitoring Active")
        else:
            st.warning("⚠️ Monitoring Unavailable")

def render_patient_overview_tab():
    """Render the Patient Overview tab with dynamic data"""
    st.header("👤 Patient Demographics & Health Metrics")
    
    if not st.session_state.patient_data:
        st.error("No patient data available. Please log in again.")
        return
    
    patient = st.session_state.patient_data
    demographics = patient['demographics']
    lab_results = patient['lab_results']
    lifestyle = patient['lifestyle_data']
    
    # Patient Demographics
    st.subheader("📋 Demographics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{patient['age']} years")
        st.metric("Gender", patient['gender'])
    
    with col2:
        st.metric("Height", f"{demographics['height']} cm")
        st.metric("Weight", f"{demographics['weight']} kg")
    
    with col3:
        st.metric("BMI", f"{demographics['bmi']:.1f}")
        st.metric("Blood Type", demographics['blood_type'])
    
    with col4:
        st.metric("Phone", demographics['phone'])
        st.info(f"**Emergency Contact:** {demographics['emergency_contact']}")
    
    # Lab Results
    st.subheader("🧪 Latest Lab Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cholesterol", f"{lab_results['cholesterol']} mg/dL", 
                 delta=None, delta_color="inverse" if lab_results['cholesterol'] > 240 else "normal")
        st.metric("HDL", f"{lab_results['hdl']} mg/dL")
        st.metric("LDL", f"{lab_results['ldl']} mg/dL")
    
    with col2:
        st.metric("Triglycerides", f"{lab_results['triglycerides']} mg/dL")
        st.metric("Glucose", f"{lab_results['glucose']} mg/dL")
        st.metric("HbA1c", f"{lab_results['hba1c']}%")
    
    with col3:
        st.metric("Systolic BP", f"{lab_results['systolic_bp']} mmHg")
        st.metric("Diastolic BP", f"{lab_results['diastolic_bp']} mmHg")
        st.info(f"**Last Test:** {lab_results['last_test_date']}")
    
    # Lifestyle Data
    st.subheader("🏃 Lifestyle Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Editable lifestyle data with session state persistence
        smoking_status = st.selectbox(
            "Smoking Status", 
            ["Never", "Former", "Current"],
            index=["Never", "Former", "Current"].index(lifestyle['smoking_status']),
            key="smoking_status_select"
        )
        
        alcohol_consumption = st.selectbox(
            "Alcohol Consumption",
            ["None", "Light", "Moderate", "Heavy"],
            index=["None", "Light", "Moderate", "Heavy"].index(lifestyle['alcohol_consumption']),
            key="alcohol_select"
        )
    
    with col2:
        exercise_frequency = st.number_input(
            "Exercise Frequency (days/week)",
            min_value=0, max_value=7,
            value=lifestyle['exercise_frequency'],
            key="exercise_input"
        )
        
        diet_type = st.selectbox(
            "Diet Type",
            ["Standard", "Mediterranean", "Vegetarian", "Vegan", "Low-carb"],
            index=["Standard", "Mediterranean", "Vegetarian", "Vegan", "Low-carb"].index(lifestyle['diet_type']),
            key="diet_select"
        )
    
    with col3:
        stress_level = st.selectbox(
            "Stress Level",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(lifestyle['stress_level']),
            key="stress_select"
        )
        
        sleep_hours = st.number_input(
            "Sleep Hours/night",
            min_value=0, max_value=12,
            value=lifestyle['sleep_hours'],
            key="sleep_input"
        )
    
    # Update patient data if changes were made
    if (smoking_status != lifestyle['smoking_status'] or 
        alcohol_consumption != lifestyle['alcohol_consumption'] or
        exercise_frequency != lifestyle['exercise_frequency'] or
        diet_type != lifestyle['diet_type'] or
        stress_level != lifestyle['stress_level'] or
        sleep_hours != lifestyle['sleep_hours']):
        
        # Update session state with new values
        st.session_state.patient_data['lifestyle_data'].update({
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption,
            'exercise_frequency': exercise_frequency,
            'diet_type': diet_type,
            'stress_level': stress_level,
            'sleep_hours': sleep_hours
        })
        
        st.success("✅ Lifestyle data updated in session")
    
    # Family History
    family_history = st.checkbox(
        "Family History of Cardiovascular Disease",
        value=lifestyle['family_history'],
        key="family_history_check"
    )
    
    if family_history != lifestyle['family_history']:
        st.session_state.patient_data['lifestyle_data']['family_history'] = family_history
        st.success("✅ Family history updated in session")

def render_risk_prediction_tab():
    """Render the Risk Prediction tab with API integration"""
    st.header("🎯 Cardiovascular Risk Assessment")
    
    if not st.session_state.patient_data:
        st.error("No patient data available. Please log in again.")
        return
    
    patient = st.session_state.patient_data
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Calculate Risk Score", type="primary", use_container_width=True):
            with st.spinner("Calculating risk score..."):
                result = api_client.predict_patient_risk(patient['id'])
                
                if result["status"] == "success":
                    st.session_state.prediction_results = result["data"]
                    st.success("✅ Risk assessment completed!")
                else:
                    st.error(f"❌ Prediction failed: {result['message']}")
    
    # Display results if available
    if st.session_state.prediction_results:
        results = st.session_state.prediction_results
        
        # Risk score display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_score = results['risk_score']
            risk_category = results['risk_category']
            
            if risk_category == "High":
                risk_color = "red"
            elif risk_category == "Medium":
                risk_color = "orange"
            else:
                risk_color = "green"
            
            st.markdown(f"""
            <div class="metric-card risk-{risk_category.lower()}">
                <h3>Risk Score</h3>
                <h2 style="color: {risk_color}">{risk_score:.2%}</h2>
                <p>Risk Level: <strong>{risk_category}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_interval = results['confidence_interval']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence Interval</h3>
                <h4>{confidence_interval['lower']:.2%} - {confidence_interval['upper']:.2%}</h4>
                <p>Model Version: {results['model_version']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            prediction_time = results['prediction_timestamp']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction Time</h3>
                <p>{prediction_time}</p>
                <p>Patient ID: {results['patient_id']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # SHAP values visualization
        st.subheader("📊 Risk Factor Analysis (SHAP Values)")
        
        shap_values = results['shap_values']
        factors = list(shap_values.keys())
        values = list(shap_values.values())
        
        # Create SHAP bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=factors,
                orientation='h',
                marker_color=['red' if v > 0 else 'blue' for v in values]
            )
        ])
        
        fig.update_layout(
            title="Feature Importance (SHAP Values)",
            xaxis_title="Impact on Risk Score",
            yaxis_title="Risk Factors",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors explanation
        st.subheader("🔍 Risk Factors Explanation")
        
        for factor, value in shap_values.items():
            impact = "increases" if value > 0 else "decreases"
            color = "🔴" if value > 0 else "🔵"
            st.write(f"{color} **{factor.replace('_', ' ').title()}**: {impact} risk by {abs(value):.3f}")

def render_recommendations_tab():
    """Render the Recommendations tab with API integration"""
    st.header("📋 Personalized Healthcare Recommendations")
    
    if not st.session_state.patient_data:
        st.error("No patient data available. Please log in again.")
        return
    
    patient = st.session_state.patient_data
    
    # Get recommendations button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📋 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                result = api_client.get_patient_recommendations(patient['id'])
                
                if result["status"] == "success":
                    st.session_state.recommendations = result["data"]
                    st.success("✅ Recommendations generated!")
                else:
                    st.error(f"❌ Failed to get recommendations: {result['message']}")
    
    # Display recommendations if available
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        
        # Risk level summary
        st.subheader(f"🎯 Risk Level: {recommendations['risk_level']}")
        
        # Alerts (if any)
        if recommendations.get('alerts'):
            st.subheader("🚨 Important Alerts")
            for alert in recommendations['alerts']:
                st.error(f"⚠️ {alert}")
        
        # Doctor guidance
        if recommendations.get('doctor_guidance'):
            st.subheader("👨‍⚕️ Doctor Guidance")
            for guidance in recommendations['doctor_guidance']:
                st.info(f"• {guidance}")
        
        # Lifestyle recommendations
        if recommendations.get('lifestyle_recommendations'):
            st.subheader("🏃 Lifestyle Recommendations")
            for rec in recommendations['lifestyle_recommendations']:
                st.success(f"• {rec}")
        
        # Medication suggestions
        if recommendations.get('medication_suggestions'):
            st.subheader("💊 Medication Suggestions")
            for med in recommendations['medication_suggestions']:
                st.warning(f"• {med}")
        
        # Follow-up schedule
        if recommendations.get('follow_up_schedule'):
            st.subheader("📅 Follow-up Schedule")
            schedule = recommendations['follow_up_schedule']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Next Appointment", schedule.get('next_appointment', 'N/A'))
            with col2:
                st.metric("Lab Recheck", schedule.get('lab_recheck', 'N/A'))
            with col3:
                st.metric("Imaging", schedule.get('imaging', 'N/A'))
        
        # Export recommendations
        st.subheader("📤 Export Recommendations")
        
        # Create export data
        export_data = {
            "patient_id": patient['id'],
            "patient_name": patient['name'],
            "risk_level": recommendations['risk_level'],
            "generated_at": recommendations['generated_at'],
            "recommendations": recommendations
        }
        
        export_json = json.dumps(export_data, indent=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download as JSON",
                data=export_json,
                file_name=f"recommendations_{patient['id']}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col2:
            # Create a simple text summary for download
            text_summary = f"""Healthcare Recommendations for {patient['name']} (ID: {patient['id']})
Generated: {recommendations['generated_at']}
Risk Level: {recommendations['risk_level']}

Doctor Guidance:
{chr(10).join(['- ' + g for g in recommendations.get('doctor_guidance', [])])}

Lifestyle Recommendations:
{chr(10).join(['- ' + r for r in recommendations.get('lifestyle_recommendations', [])])}

Medication Suggestions:
{chr(10).join(['- ' + m for m in recommendations.get('medication_suggestions', [])])}
"""
            
            st.download_button(
                label="📝 Download as Text",
                data=text_summary,
                file_name=f"recommendations_{patient['id']}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

def render_cohort_view_tab():
    """Render cohort risk analysis tab"""
    st.header("👥 Cohort Risk Analysis")
    
    try:
        # Fetch cohort risk data from Supabase
        cohort_data = get_cohort_risk_data()
        
        if cohort_data:
            
            # Display summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Patients", cohort_data["total_patients"])
            
            with col2:
                st.metric("High Risk", cohort_data["high_risk_count"], 
                         delta=f"{cohort_data['high_risk_count']/cohort_data['total_patients']*100:.1f}%")
            
            with col3:
                st.metric("Medium Risk", cohort_data["medium_risk_count"],
                         delta=f"{cohort_data['medium_risk_count']/cohort_data['total_patients']*100:.1f}%")
            
            with col4:
                st.metric("Low Risk", cohort_data["low_risk_count"],
                         delta=f"{cohort_data['low_risk_count']/cohort_data['total_patients']*100:.1f}%")
            
            with col5:
                st.metric("Avg Risk Score", f"{cohort_data['average_risk_score']:.3f}")
            
            st.divider()
            
            # Risk distribution chart
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Risk Distribution")
                risk_counts = {
                    "High": cohort_data["high_risk_count"],
                    "Medium": cohort_data["medium_risk_count"],
                    "Low": cohort_data["low_risk_count"]
                }
                
                fig_pie = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    color_discrete_map={
                        "High": "#ff4444",
                        "Medium": "#ffaa00",
                        "Low": "#44aa44"
                    },
                    title="Patient Risk Categories"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Risk Score Distribution")
                risk_scores = [patient["risk_score"] for patient in cohort_data["patients"]]
                
                fig_hist = px.histogram(
                    x=risk_scores,
                    nbins=20,
                    title="Distribution of Risk Scores",
                    labels={"x": "Risk Score", "y": "Number of Patients"}
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            st.divider()
            
            # Patient risk table
            st.subheader("Patient Risk Scores")
            
            # Convert to DataFrame for better display
            df_patients = pd.DataFrame(cohort_data["patients"])
            
            # Add risk score formatting and styling
            def style_risk_score(val):
                if val >= 0.7:
                    return 'background-color: #ffebee; color: #d32f2f; font-weight: bold'
                elif val >= 0.4:
                    return 'background-color: #fff3e0; color: #f57c00; font-weight: bold'
                else:
                    return 'background-color: #e8f5e8; color: #388e3c; font-weight: bold'
            
            def style_risk_category(val):
                if val == "High":
                    return 'background-color: #ffebee; color: #d32f2f; font-weight: bold'
                elif val == "Medium":
                    return 'background-color: #fff3e0; color: #f57c00; font-weight: bold'
                else:
                    return 'background-color: #e8f5e8; color: #388e3c; font-weight: bold'
            
            # Display styled dataframe
            styled_df = df_patients.style.applymap(style_risk_score, subset=['risk_score']) \
                                        .applymap(style_risk_category, subset=['risk_category']) \
                                        .format({'risk_score': '{:.3f}'})
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Export functionality
            csv_data = df_patients.to_csv(index=False)
            st.download_button(
                label="📊 Download Cohort Data as CSV",
                data=csv_data,
                file_name=f"cohort_risk_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.error(f"Failed to fetch cohort data: {response.status_code}")
            
    except Exception as e:
        st.error(f"Error loading cohort data: {str(e)}")
        st.info("Please ensure the API server is running and accessible.")

def render_model_evaluation_tab():
    """Render model evaluation metrics tab"""
    st.header("📊 Model Performance Evaluation")
    
    try:
        # Fetch model evaluation data
        eval_data = get_model_evaluation_data()
        
        if eval_data:
            
            # Display model information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Name", eval_data["model_name"])
            
            with col2:
                st.metric("Model Version", eval_data.get("model_version", "N/A"))
            
            with col3:
                st.metric("Sample Size", eval_data["sample_size"])
            
            with col4:
                st.metric("Evaluation Date", eval_data["evaluation_date"][:10])
            
            st.divider()
            
            # Display key metrics
            st.subheader("🎯 Key Performance Metrics")
            
            metrics = eval_data["metrics"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                auroc = metrics.get("auroc", 0)
                st.metric(
                    "AUROC", 
                    f"{auroc:.3f}",
                    delta="Excellent" if auroc > 0.8 else "Good" if auroc > 0.7 else "Fair"
                )
            
            with col2:
                auprc = metrics.get("auprc", 0)
                st.metric(
                    "AUPRC", 
                    f"{auprc:.3f}",
                    delta="Excellent" if auprc > 0.8 else "Good" if auprc > 0.7 else "Fair"
                )
            
            with col3:
                sensitivity = metrics.get("sensitivity", 0)
                st.metric(
                    "Sensitivity (Recall)", 
                    f"{sensitivity:.3f}",
                    delta="High" if sensitivity > 0.8 else "Medium" if sensitivity > 0.6 else "Low"
                )
            
            with col4:
                specificity = metrics.get("specificity", 0)
                st.metric(
                    "Specificity", 
                    f"{specificity:.3f}",
                    delta="High" if specificity > 0.8 else "Medium" if specificity > 0.6 else "Low"
                )
            
            st.divider()
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                precision = metrics.get("precision", 0)
                st.metric("Precision", f"{precision:.3f}")
            
            with col2:
                f1_score = metrics.get("f1_score", 0)
                st.metric("F1 Score", f"{f1_score:.3f}")
            
            with col3:
                calibration_error = metrics.get("calibration_error", 0)
                st.metric("Calibration Error", f"{calibration_error:.3f}")
            
            with col4:
                positive_rate = eval_data["positive_cases"] / eval_data["sample_size"]
                st.metric("Positive Rate", f"{positive_rate:.1%}")
            
            st.divider()
            
            # Model interpretation
            st.subheader("🔍 Model Interpretation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Clinical Performance:**")
                if auroc > 0.8:
                    st.success("🟢 Excellent discriminative ability (AUROC > 0.8)")
                elif auroc > 0.7:
                    st.warning("🟡 Good discriminative ability (AUROC > 0.7)")
                else:
                    st.error("🔴 Fair discriminative ability (AUROC ≤ 0.7)")
                
                if sensitivity > 0.8:
                    st.success("🟢 High sensitivity - Good at detecting deterioration")
                elif sensitivity > 0.6:
                    st.warning("🟡 Moderate sensitivity - May miss some cases")
                else:
                    st.error("🔴 Low sensitivity - Missing many deterioration cases")
            
            with col2:
                st.markdown("**Model Reliability:**")
                if calibration_error < 0.1:
                    st.success("🟢 Well-calibrated predictions")
                elif calibration_error < 0.2:
                    st.warning("🟡 Moderately calibrated predictions")
                else:
                    st.error("🔴 Poorly calibrated predictions")
                
                if specificity > 0.8:
                    st.success("🟢 High specificity - Low false alarm rate")
                elif specificity > 0.6:
                    st.warning("🟡 Moderate specificity - Some false alarms")
                else:
                    st.error("🔴 Low specificity - High false alarm rate")
            
            # Recommendations based on metrics
            st.subheader("💡 Clinical Recommendations")
            
            recommendations = []
            
            if sensitivity < 0.7:
                recommendations.append("⚠️ Consider lowering the risk threshold to catch more deterioration cases")
            
            if specificity < 0.7:
                recommendations.append("⚠️ Consider raising the risk threshold to reduce false alarms")
            
            if calibration_error > 0.15:
                recommendations.append("⚠️ Model predictions may need recalibration for clinical use")
            
            if auroc < 0.75:
                recommendations.append("⚠️ Consider feature engineering or model retraining to improve performance")
            
            if not recommendations:
                recommendations.append("✅ Model performance is within acceptable clinical ranges")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
        else:
            st.error(f"Failed to fetch model evaluation: {response.status_code}")
            
    except Exception as e:
        st.error(f"Error loading model evaluation: {str(e)}")
        st.info("Please ensure the API server is running and accessible.")

def render_explainability_tab():
    """Render the Model Explainability tab"""
    st.header("🔍 Model Explainability")
    st.markdown("""Understand how the AI model makes predictions using SHAP (SHapley Additive exPlanations) 
    to provide transparent and interpretable risk assessments.""")
    
    # Explainability options
    explainability_type = st.radio(
        "Select Explanation Type:",
        ["Global Explanations", "Patient-Specific Explanations"],
        horizontal=True
    )
    
    if explainability_type == "Global Explanations":
        st.subheader("🌍 Global Model Explanations")
        st.markdown("Understanding which factors are most important across all patients for 90-day deterioration risk.")
        
        if st.button("Generate Global Explanations", type="primary"):
            with st.spinner("Generating global model explanations..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/explain/global")
                    if response.status_code == 200:
                        global_data = response.json()
                        
                        # Display global insights
                        st.success("✅ Global explanations generated successfully!")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📊 Feature Importance Ranking")
                            
                            # Extract feature contributions
                            feature_contributions = global_data.get("feature_importance", [])
                            if feature_contributions:
                                # Create DataFrame for visualization
                                df_features = pd.DataFrame(feature_contributions)
                                df_features = df_features.sort_values('abs_shap_value', ascending=False).head(10)
                                
                                # Create horizontal bar chart
                                fig_importance = px.bar(
                                    df_features,
                                    x='abs_shap_value',
                                    y='feature',
                                    orientation='h',
                                    color='contribution_type',
                                    color_discrete_map={
                                        'increases_risk': '#ff4444',
                                        'decreases_risk': '#44aa44'
                                    },
                                    title="Top 10 Most Important Features",
                                    labels={
                                        'abs_shap_value': 'Feature Importance (|SHAP Value|)',
                                        'feature': 'Clinical Features',
                                        'contribution_type': 'Impact Type'
                                    }
                                )
                                fig_importance.update_layout(height=500)
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Feature details table
                                st.subheader("📋 Detailed Feature Analysis")
                                display_df = df_features[['feature', 'description', 'abs_shap_value', 'impact_strength', 'contribution_type']].copy()
                                display_df.columns = ['Feature', 'Clinical Meaning', 'Importance Score', 'Impact Level', 'Effect on Risk']
                                display_df['Importance Score'] = display_df['Importance Score'].round(4)
                                st.dataframe(display_df, use_container_width=True)
                        
                        with col2:
                            st.subheader("🎯 Key Insights")
                            
                            model_insights = global_data.get("model_insights", {})
                            
                            # Top risk factors
                            st.markdown("**🔴 Top Risk Drivers:**")
                            top_factors = model_insights.get("most_important_factors", [])
                            for i, factor in enumerate(top_factors[:5], 1):
                                st.markdown(f"{i}. **{factor.get('feature', 'N/A')}** (Impact: {factor.get('impact_strength', 'N/A')})")
                            
                            # Protective factors
                            st.markdown("**🟢 Protective Factors:**")
                            protective_factors = model_insights.get("protective_factors", [])
                            for i, factor in enumerate(protective_factors[:5], 1):
                                st.markdown(f"{i}. **{factor.get('feature', 'N/A')}** (Protection: {factor.get('impact_strength', 'N/A')})")
                            
                            # Clinical interpretation
                            st.markdown("**🏥 Clinical Interpretation:**")
                            st.info(model_insights.get("clinical_interpretation", "No interpretation available."))
                    
                    else:
                        st.error(f"❌ Failed to generate global explanations: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error generating global explanations: {str(e)}")
    
    else:  # Patient-Specific Explanations
        st.subheader("👤 Patient-Specific Explanations")
        st.markdown("Get detailed explanations for why a specific patient has their current risk score.")
        
        # Patient selection
        try:
            patients_response = requests.get(f"{API_BASE_URL}/patients")
            if patients_response.status_code == 200:
                patients_data = patients_response.json()
                patients = patients_data.get("patients", [])
                
                if patients:
                    # Create patient selection dropdown
                    patient_options = {f"{p['name']} (ID: {p['id']}) - Age {p['age']}": p['id'] for p in patients}
                    selected_patient_display = st.selectbox(
                        "Select a patient for detailed explanation:",
                        options=list(patient_options.keys())
                    )
                    selected_patient_id = patient_options[selected_patient_display]
                    
                    if st.button("Generate Patient Explanation", type="primary"):
                        with st.spinner(f"Generating explanation for {selected_patient_display.split(' (')[0]}..."):
                            try:
                                response = requests.post(f"{API_BASE_URL}/explain/local/{selected_patient_id}")
                                if response.status_code == 200:
                                    local_data = response.json()
                                    
                                    st.success("✅ Patient explanation generated successfully!")
                                    
                                    # Patient info and prediction
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    
                                    prediction = local_data.get("prediction", {})
                                    
                                    with col1:
                                        st.metric(
                                            "Risk Score", 
                                            f"{prediction.get('risk_score', 0):.3f}",
                                            delta=f"Category: {prediction.get('risk_category', 'Unknown')}"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "Risk Category", 
                                            prediction.get('risk_category', 'Unknown'),
                                            delta=f"Model: {prediction.get('model_version', 'v1.0')}"
                                        )
                                    
                                    with col3:
                                        recommendations = local_data.get("clinical_recommendations", {})
                                        st.metric(
                                            "Monitoring Priority", 
                                            recommendations.get('monitoring_priority', 'Medium')
                                        )
                                    
                                    st.divider()
                                    
                                    # Risk drivers and protective factors
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.subheader("🔴 Risk Drivers")
                                        risk_drivers = local_data.get("risk_drivers", [])
                                        
                                        if risk_drivers:
                                            for i, driver in enumerate(risk_drivers[:5], 1):
                                                with st.expander(f"{i}. {driver.get('feature', 'Unknown')} (Impact: {driver.get('impact_strength', 'N/A')})"):
                                                    st.markdown(f"**Current Value:** {driver.get('formatted_value', 'N/A')}")
                                                    st.markdown(f"**SHAP Contribution:** {driver.get('shap_value', 0):.4f}")
                                                    st.markdown(f"**Clinical Meaning:** {driver.get('description', 'No description available')}")
                                        else:
                                            st.info("No significant risk drivers identified.")
                                    
                                    with col2:
                                        st.subheader("🟢 Protective Factors")
                                        protective_factors = local_data.get("protective_factors", [])
                                        
                                        if protective_factors:
                                            for i, factor in enumerate(protective_factors[:5], 1):
                                                with st.expander(f"{i}. {factor.get('feature', 'Unknown')} (Protection: {factor.get('impact_strength', 'N/A')})"):
                                                    st.markdown(f"**Current Value:** {factor.get('formatted_value', 'N/A')}")
                                                    st.markdown(f"**SHAP Contribution:** {factor.get('shap_value', 0):.4f}")
                                                    st.markdown(f"**Clinical Meaning:** {factor.get('description', 'No description available')}")
                                        else:
                                            st.info("No significant protective factors identified.")
                                    
                                    # Clinical recommendations
                                    st.subheader("🏥 Clinical Recommendations")
                                    recommendations = local_data.get("clinical_recommendations", {})
                                    
                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        st.info(f"**High Impact Factors:** {recommendations.get('high_impact_factors', 'N/A')}")
                                    
                                    with col2:
                                        st.info(f"**Actionable Insights:** {recommendations.get('actionable_insights', 'N/A')}")
                                
                                else:
                                    st.error(f"❌ Failed to generate patient explanation: {response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error generating patient explanation: {str(e)}")
                
                else:
                    st.warning("⚠️ No patients available for explanation.")
            
            else:
                st.error("❌ Failed to load patient data.")
                
        except Exception as e:
            st.error(f"❌ Error loading patients: {str(e)}")
    
    # Educational content
    st.divider()
    st.subheader("📚 Understanding SHAP Explanations")
    
    with st.expander("What are SHAP values?"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** values provide a unified framework for interpreting model predictions:
        
        - **Positive SHAP values** increase the risk prediction
        - **Negative SHAP values** decrease the risk prediction  
        - **Magnitude** indicates the strength of the feature's impact
        - **Sum of all SHAP values** + base value = final prediction
        
        This helps clinicians understand exactly which factors contribute to a patient's risk score.
        """)
    
    with st.expander("How to interpret the results?"):
        st.markdown("""
        **For Clinical Decision Making:**
        
        1. **Focus on high-impact factors** - These have the strongest influence on the prediction
        2. **Consider modifiable risk factors** - Target interventions on factors that can be changed
        3. **Monitor protective factors** - Ensure positive factors are maintained
        4. **Use for patient education** - Help patients understand their specific risk factors
        5. **Validate with clinical judgment** - AI explanations should complement, not replace, clinical expertise
        """)

def get_cohort_risk_data() -> Dict:
    """Get cohort risk analysis data from Supabase"""
    try:
        # Get all patients with basic info
        patients_result = supabase_client.client.table("patients").select("id, name, age, gender, bp, cholesterol, diabetes_status").execute()
        
        if not patients_result.data:
            return None
        
        patients = patients_result.data
        total_patients = len(patients)
        
        # Calculate risk categories based on available data
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        risk_scores = []
        
        for patient in patients:
            # Simple risk calculation based on available fields
            risk_score = 0.0
            
            # Age factor
            age = patient.get('age', 0)
            if age > 65:
                risk_score += 0.3
            elif age > 50:
                risk_score += 0.2
            elif age > 35:
                risk_score += 0.1
            
            # BP factor
            bp = patient.get('bp', 'normal')
            if bp == 'high':
                risk_score += 0.3
            elif bp == 'elevated':
                risk_score += 0.2
            
            # Cholesterol factor
            cholesterol = patient.get('cholesterol', 'normal')
            if cholesterol == 'high':
                risk_score += 0.2
            elif cholesterol == 'borderline':
                risk_score += 0.1
            
            # Diabetes factor
            diabetes = patient.get('diabetes_status', 'none')
            if diabetes == 'type2':
                risk_score += 0.3
            elif diabetes == 'prediabetes':
                risk_score += 0.2
            
            risk_scores.append(risk_score)
            
            # Categorize risk
            if risk_score >= 0.7:
                high_risk += 1
            elif risk_score >= 0.4:
                medium_risk += 1
            else:
                low_risk += 1
        
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        return {
            "total_patients": total_patients,
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "average_risk_score": avg_risk_score,
            "risk_scores": risk_scores,
            "patients": patients
        }
    except Exception as e:
        st.error(f"Error fetching cohort data: {e}")
        return None

def get_model_evaluation_data() -> Dict:
    """Get model evaluation data (mock data for now)"""
    try:
        # Since we don't have model evaluation tables yet, return mock data
        return {
            "model_name": "XGBoost Cardiovascular Risk Predictor",
            "model_version": "v1.2.0",
            "sample_size": 1000,
            "evaluation_date": "2024-01-15T10:30:00Z",
            "metrics": {
                "auroc": 0.847,
                "auprc": 0.723,
                "accuracy": 0.812,
                "precision": 0.789,
                "recall": 0.756,
                "f1_score": 0.772,
                "specificity": 0.834
            },
            "confusion_matrix": {
                "true_positives": 151,
                "false_positives": 33,
                "true_negatives": 667,
                "false_negatives": 49
            },
            "feature_importance": [
                {"feature": "age", "importance": 0.234},
                {"feature": "cholesterol_total", "importance": 0.187},
                {"feature": "bp_systolic", "importance": 0.156},
                {"feature": "diabetes_status", "importance": 0.143},
                {"feature": "smoking_status", "importance": 0.098}
            ]
        }
    except Exception as e:
        st.error(f"Error fetching model evaluation data: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        render_login_screen()
        return
    
    # Render main dashboard
    st.markdown('<h1 class="main-header">🏥 Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👤 Patient Overview", "🎯 Risk Prediction", "📋 Recommendations", "👥 Cohort View", "📊 Model Evaluation", "🔍 Model Explainability"])
    
    with tab1:
        render_patient_overview_tab()
    
    with tab2:
        render_risk_prediction_tab()
    
    with tab3:
        render_recommendations_tab()
    
    with tab4:
        render_cohort_view_tab()
    
    with tab5:
        render_model_evaluation_tab()
    
    with tab6:
        render_explainability_tab()

if __name__ == "__main__":
    main()