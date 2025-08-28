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

# API Configuration
API_BASE_URL = "http://localhost:8000"

class APIClient:
    """Client for interacting with FastAPI backend"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_health_status(self) -> Dict:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return {"status": "healthy", "data": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patients(self) -> Dict:
        """Get list of all patients"""
        try:
            response = self.session.get(f"{self.base_url}/patients", timeout=10)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patient_data(self, patient_id: str) -> Dict:
        """Get detailed patient data"""
        try:
            response = self.session.get(f"{self.base_url}/patients/{patient_id}", timeout=10)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_patient_risk(self, patient_id: str) -> Dict:
        """Get risk prediction for specific patient"""
        try:
            response = self.session.get(f"{self.base_url}/predict/{patient_id}", timeout=10)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_patient_recommendations(self, patient_id: str) -> Dict:
        """Get recommendations for specific patient"""
        try:
            response = self.session.get(f"{self.base_url}/recommendations/{patient_id}", timeout=10)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_monitoring_status(self) -> Dict:
        """Get monitoring system status"""
        try:
            response = self.session.get(f"{self.base_url}/monitoring/status", timeout=5)
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            else:
                return {"status": "error", "message": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Initialize API client
api_client = APIClient(API_BASE_URL)

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
    tab1, tab2, tab3 = st.tabs(["👤 Patient Overview", "🎯 Risk Prediction", "📋 Recommendations"])
    
    with tab1:
        render_patient_overview_tab()
    
    with tab2:
        render_risk_prediction_tab()
    
    with tab3:
        render_recommendations_tab()

if __name__ == "__main__":
    main()