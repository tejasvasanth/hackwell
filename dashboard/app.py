import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.success-card {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
}
.warning-card {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
}
.error-card {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)

def get_api_data(endpoint: str) -> Dict[str, Any]:
    """Fetch data from FastAPI backend"""
    try:
        response = requests.get(f"http://localhost:8000{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return {}
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return {}

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # Model performance metrics
    performance_data = pd.DataFrame({
        'date': dates,
        'r2_score': 0.85 + np.random.normal(0, 0.02, len(dates)),
        'rmse': 0.12 + np.random.normal(0, 0.01, len(dates)),
        'predictions_count': np.random.randint(100, 500, len(dates))
    })
    
    # Feature importance data
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(10)],
        'importance': np.random.exponential(0.1, 10)
    }).sort_values('importance', ascending=False)
    
    # Prediction distribution
    predictions = np.random.normal(0, 1, 1000)
    
    return performance_data, feature_importance, predictions

def main():
    """Main dashboard application"""
    st.title("🤖 ML Pipeline Dashboard")
    st.markdown("Monitor your machine learning pipeline performance and model insights")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Model Performance", "Predictions", "Feature Analysis", "Pipeline Status"]
    )
    
    # Get sample data
    performance_data, feature_importance, predictions = create_sample_data()
    
    if page == "Overview":
        show_overview(performance_data, feature_importance)
    elif page == "Model Performance":
        show_model_performance(performance_data)
    elif page == "Predictions":
        show_predictions(predictions)
    elif page == "Feature Analysis":
        show_feature_analysis(feature_importance)
    elif page == "Pipeline Status":
        show_pipeline_status()

def show_overview(performance_data: pd.DataFrame, feature_importance: pd.DataFrame):
    """Show overview dashboard"""
    st.header("📊 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_r2 = performance_data['r2_score'].iloc[-1]
        st.metric(
            label="Current R² Score",
            value=f"{current_r2:.3f}",
            delta=f"{current_r2 - performance_data['r2_score'].iloc[-2]:.3f}"
        )
    
    with col2:
        current_rmse = performance_data['rmse'].iloc[-1]
        st.metric(
            label="Current RMSE",
            value=f"{current_rmse:.3f}",
            delta=f"{current_rmse - performance_data['rmse'].iloc[-2]:.3f}",
            delta_color="inverse"
        )
    
    with col3:
        total_predictions = performance_data['predictions_count'].sum()
        st.metric(
            label="Total Predictions",
            value=f"{total_predictions:,}",
            delta="+150 today"
        )
    
    with col4:
        st.metric(
            label="Model Version",
            value="v1.2.3",
            delta="Updated 2h ago"
        )
    
    # Performance trend
    st.subheader("Performance Trend")
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('R² Score Over Time', 'RMSE Over Time'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=performance_data['date'],
            y=performance_data['r2_score'],
            mode='lines+markers',
            name='R² Score',
            line=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=performance_data['date'],
            y=performance_data['rmse'],
            mode='lines+markers',
            name='RMSE',
            line=dict(color='#ff7f0e')
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Features")
        fig = px.bar(
            feature_importance.head(5),
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance (Top 5)"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("System Health")
        
        # Health status cards
        st.markdown("""
        <div class="metric-card success-card">
            <h4>✅ API Status</h4>
            <p>All endpoints responding normally</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card success-card">
            <h4>✅ Model Status</h4>
            <p>Model loaded and serving predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card warning-card">
            <h4>⚠️ Data Drift</h4>
            <p>Minor drift detected in feature_3</p>
        </div>
        """, unsafe_allow_html=True)

def show_model_performance(performance_data: pd.DataFrame):
    """Show detailed model performance metrics"""
    st.header("📈 Model Performance")
    
    # Performance metrics over time
    fig = px.line(
        performance_data,
        x='date',
        y=['r2_score', 'rmse'],
        title="Model Performance Metrics Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction volume
    fig = px.bar(
        performance_data,
        x='date',
        y='predictions_count',
        title="Daily Prediction Volume"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance statistics
    st.subheader("Performance Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**R² Score Statistics**")
        st.write(performance_data['r2_score'].describe())
    
    with col2:
        st.write("**RMSE Statistics**")
        st.write(performance_data['rmse'].describe())

def show_predictions(predictions: np.ndarray):
    """Show prediction analysis and testing interface"""
    st.header("🎯 Predictions")
    
    # Prediction interface
    st.subheader("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Features**")
        features = {}
        for i in range(5):
            features[f'feature_{i}'] = st.number_input(
                f'Feature {i}',
                value=0.0,
                step=0.1,
                key=f'feature_{i}'
            )
    
    with col2:
        if st.button("Get Prediction", type="primary"):
            # Simulate API call
            prediction = np.random.normal(0, 1)
            st.success(f"Prediction: {prediction:.3f}")
            
            # Show explanation (placeholder)
            st.write("**Feature Contributions:**")
            for feature, value in features.items():
                contribution = np.random.normal(0, 0.1)
                st.write(f"- {feature}: {contribution:.3f}")
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    fig = px.histogram(
        x=predictions,
        nbins=50,
        title="Distribution of Recent Predictions"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis(feature_importance: pd.DataFrame):
    """Show feature analysis and importance"""
    st.header("🔍 Feature Analysis")
    
    # Feature importance
    fig = px.bar(
        feature_importance,
        x='feature',
        y='importance',
        title="Feature Importance Ranking"
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation heatmap (sample data)
    st.subheader("Feature Correlation Matrix")
    np.random.seed(42)
    corr_matrix = np.random.rand(10, 10)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1)  # Set diagonal to 1
    
    fig = px.imshow(
        corr_matrix,
        x=[f'feature_{i}' for i in range(10)],
        y=[f'feature_{i}' for i in range(10)],
        color_continuous_scale='RdBu',
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_pipeline_status():
    """Show pipeline status and logs"""
    st.header("⚙️ Pipeline Status")
    
    # Pipeline runs
    st.subheader("Recent Pipeline Runs")
    
    pipeline_data = pd.DataFrame({
        'Pipeline': ['Training', 'Data Ingestion', 'Monitoring', 'Training', 'Data Ingestion'],
        'Status': ['✅ Success', '✅ Success', '⚠️ Warning', '✅ Success', '❌ Failed'],
        'Start Time': pd.date_range(start='2024-01-30', periods=5, freq='6H'),
        'Duration': ['12m 34s', '2m 15s', '5m 42s', '11m 58s', '1m 23s'],
        'Records': [1000, 5000, 0, 1200, 0]
    })
    
    st.dataframe(pipeline_data, use_container_width=True)
    
    # System logs
    st.subheader("System Logs")
    
    log_entries = [
        "[2024-01-31 10:30:15] INFO: Model training completed successfully",
        "[2024-01-31 10:25:42] INFO: Data validation passed",
        "[2024-01-31 10:20:18] WARNING: Minor data drift detected in feature_3",
        "[2024-01-31 10:15:33] INFO: New data batch ingested (5000 records)",
        "[2024-01-31 10:10:07] INFO: Model serving 150 predictions/hour"
    ]
    
    for log in log_entries:
        if "ERROR" in log or "Failed" in log:
            st.error(log)
        elif "WARNING" in log or "Warning" in log:
            st.warning(log)
        else:
            st.info(log)

if __name__ == "__main__":
    main()