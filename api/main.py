from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime, timedelta
from loguru import logger

from config.settings import settings
# from ml.mlflow_service import mlflow_service
from ml.explainer import ModelExplainer
# from ml.metrics_service import metrics_service
# from ml.recommendation_service import recommendation_service
from ml.monitoring_service import monitoring_service
from ml.model import MLModel
from supabase_client import SupabaseClient
from api.models.monitoring import (
    DriftReport, MonitoringStatus, AlertsResponse, 
    MonitoringResponse, ThresholdUpdate, ReportRequest
)

app = FastAPI(
    title="ML Pipeline API",
    description="FastAPI application for ML model serving with explainability",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
explainer = ModelExplainer()
ml_model = MLModel()
supabase_client = SupabaseClient()

# @app.on_event("startup")
# async def startup_event():
#     """Initialize services on startup"""
#     logger.info("Starting FastAPI application...")
#     
#     # Initialize MLflow service
#     await mlflow_service.initialize()
#     logger.info("MLflow service initialized")
#     
#     # Initialize metrics service
#     await metrics_service.initialize()
#     logger.info("Metrics service initialized")
#     
#     logger.info("FastAPI application startup complete")

# Pydantic models
class PatientPredictionRequest(BaseModel):
    patient_id: str
    features: Optional[Dict[str, Any]] = None
    explain: bool = True
    ground_truth: Optional[float] = None  # For logging actual outcomes

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    explain: bool = False

class PredictionResponse(BaseModel):
    patient_id: Optional[str] = None
    prediction: float
    probability: float
    risk_level: str
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    model_info: Dict[str, Any]
    timestamp: datetime

class MetricsResponse(BaseModel):
    evaluation_metrics: Dict[str, Any]
    drift_analysis: Dict[str, Any]
    prediction_stats: Dict[str, Any]
    timestamp: datetime

class RecommendationResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    recommendations: List[Dict[str, Any]]
    total_recommendations: int
    generated_at: datetime
    patient_profile: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model_status: Dict[str, Any]

class PatientResponse(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    demographics: Dict[str, Any]
    lab_results: Dict[str, Any]
    lifestyle_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class PatientListResponse(BaseModel):
    patients: List[Dict[str, Any]]
    total_count: int
    timestamp: datetime

class ModelEvaluationResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    evaluation_date: str
    metrics: Dict[str, float]
    sample_size: int
    positive_cases: int
    negative_cases: int

class CohortRiskResponse(BaseModel):
    patients: List[Dict[str, Any]]
    total_patients: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    average_risk_score: float
    timestamp: datetime

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        model_status={
            "model_loaded": True,
            "model_name": "cardiovascular_risk_model",
            "model_version": "1.0.0",
            "status": "available"
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_patient(request: PatientPredictionRequest, background_tasks: BackgroundTasks):
    """Make prediction for patient with risk score and SHAP/LIME explanation"""
    try:
        # Get features from request or use default patient features
        features = request.features or {
            "age": 65, "cholesterol": 240, "blood_pressure": 140,
            "diabetes": 1, "smoking": 0, "family_history": 1
        }
        
        # Simple risk calculation based on features (mock prediction)
        risk_score = 0.0
        risk_score += features.get("age", 0) * 0.01
        risk_score += features.get("cholesterol", 0) * 0.002
        risk_score += features.get("blood_pressure", 0) * 0.003
        risk_score += features.get("diabetes", 0) * 0.3
        risk_score += features.get("smoking", 0) * 0.2
        risk_score += features.get("family_history", 0) * 0.15
        
        # Normalize to probability (0-1)
        probability = min(max(risk_score, 0.0), 1.0)
        
        # Calculate risk level and confidence
        if probability >= 0.7:
            risk_level = "High"
            confidence = probability
        elif probability >= 0.4:
            risk_level = "Medium"
            confidence = 1 - abs(0.55 - probability)
        else:
            risk_level = "Low"
            confidence = 1 - probability
        
        response_data = {
            "patient_id": request.patient_id,
            "prediction": float(probability),
            "probability": probability,
            "risk_level": risk_level,
            "confidence": float(confidence),
            "model_info": "mock_model_v1.0",
            "timestamp": datetime.now()
        }
        
        # Store prediction in Supabase
        prediction_data = {
            "patient_id": request.patient_id,
            "risk_score": float(probability),
            "risk_category": risk_level,
            "confidence": float(confidence),
            "model_version": "mock_model_v1.0",
            "features_used": features,
            "prediction_date": datetime.now().isoformat()
        }
        
        await supabase_client.create_prediction(prediction_data)
        
        # Add simple explanation if requested
        if request.explain:
            explanation = {
                "feature_importance": {
                    "age": features.get("age", 0) * 0.01,
                    "cholesterol": features.get("cholesterol", 0) * 0.002,
                    "blood_pressure": features.get("blood_pressure", 0) * 0.003,
                    "diabetes": features.get("diabetes", 0) * 0.3,
                    "smoking": features.get("smoking", 0) * 0.2,
                    "family_history": features.get("family_history", 0) * 0.15
                },
                "method": "mock_explanation"
            }
            response_data["explanation"] = explanation
        
        return PredictionResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Prediction error for patient {request.patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{patient_id}", response_model=PredictionResponse)
async def predict_patient_risk(patient_id: str):
    """Predict cardiovascular risk for a specific patient by ID"""
    try:
        # First get patient data
        patient_response = await get_patient_data(patient_id)
        patient = patient_response.dict()
        
        # Extract relevant data for prediction
        lab_results = patient["lab_results"]
        lifestyle = patient["lifestyle_data"]
        
        # Create prediction request from patient data
        request = PatientPredictionRequest(
            patient_id=patient_id,
            age=patient["age"],
            gender=patient["gender"],
            cholesterol=lab_results["cholesterol"],
            hdl=lab_results["hdl"],
            ldl=lab_results["ldl"],
            triglycerides=lab_results["triglycerides"],
            systolic_bp=lab_results["systolic_bp"],
            diastolic_bp=lab_results["diastolic_bp"],
            glucose=lab_results["glucose"],
            smoking_status=lifestyle["smoking_status"],
            exercise_frequency=lifestyle["exercise_frequency"],
            family_history=lifestyle["family_history"],
            diabetes=lab_results["hba1c"] > 6.5  # Simple diabetes check
        )
        
        # Use existing prediction logic
        return await predict_risk(request)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Patient prediction error for {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get evaluation metrics, drift analysis, and prediction statistics"""
    try:
        # Mock evaluation metrics
        evaluation_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "auc_roc": 0.91
        }
        
        # Mock drift analysis
        drift_analysis = {
            "data_drift_detected": False,
            "model_drift_detected": False,
            "drift_score": 0.15,
            "threshold": 0.3
        }
        
        # Mock prediction statistics
        prediction_stats = {
            "total_predictions": 1250,
            "high_risk_predictions": 125,
            "medium_risk_predictions": 375,
            "low_risk_predictions": 750,
            "average_confidence": 0.78
        }
        
        return MetricsResponse(
            evaluation_metrics=evaluation_metrics,
            drift_analysis=drift_analysis,
            prediction_stats=prediction_stats,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Metrics retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/{patient_id}", response_model=RecommendationResponse)
async def get_recommendations(patient_id: str):
    """Get personalized healthcare recommendations based on patient risk and profile"""
    try:
        # Get patient data and prediction
        patient_response = await get_patient_data(patient_id)
        prediction_response = await predict_patient_risk(patient_id)
        
        patient = patient_response.dict()
        prediction = prediction_response.dict()
        
        # Extract patient profile data
        lab_results = patient["lab_results"]
        lifestyle = patient["lifestyle_data"]
        demographics = patient["demographics"]
        
        patient_profile = {
            "age": patient["age"],
            "cholesterol": lab_results["cholesterol"],
            "blood_pressure": lab_results["systolic_bp"],
            "diabetes": 1 if lab_results["hba1c"] > 6.5 else 0,
            "smoking": 1 if lifestyle["smoking_status"] == "Current" else 0,
            "family_history": 1 if lifestyle["family_history"] else 0,
            "bmi": demographics["bmi"],
            "exercise_frequency": lifestyle["exercise_frequency"]
        }
        
        # Generate recommendations based on patient data and risk
        recommendations = []
        
        # Risk-based recommendations
        if prediction["risk_category"] == "High":
            recommendations.append({
                "category": "Medical",
                "recommendation": "Immediate cardiology consultation recommended",
                "priority": "High"
            })
            recommendations.append({
                "category": "Medical",
                "recommendation": "Consider stress testing and cardiac imaging",
                "priority": "High"
            })
        
        # Lifestyle-specific recommendations
        if lifestyle["smoking_status"] == "Current":
            recommendations.append({
                "category": "Lifestyle",
                "recommendation": "Smoking cessation program - highest priority",
                "priority": "High"
            })
        
        if lifestyle["exercise_frequency"] < 3:
            recommendations.append({
                "category": "Exercise",
                "recommendation": "Increase physical activity to 150 minutes/week",
                "priority": "Medium"
            })
        
        if lab_results["cholesterol"] > 240:
            recommendations.append({
                "category": "Diet",
                "recommendation": "Low-cholesterol diet consultation and consider statin therapy",
                "priority": "High" if prediction["risk_category"] == "High" else "Medium"
            })
        
        if lab_results["systolic_bp"] > 140:
            recommendations.append({
                "category": "Lifestyle",
                "recommendation": "DASH diet, sodium restriction, and blood pressure monitoring",
                "priority": "High"
            })
        
        if lab_results["hba1c"] > 6.5:
            recommendations.append({
                "category": "Medical",
                "recommendation": "Diabetes management consultation and glucose monitoring",
                "priority": "High"
            })
        
        # BMI-based recommendations
        if demographics["bmi"] > 30:
            recommendations.append({
                "category": "Lifestyle",
                "recommendation": "Weight management program for obesity",
                "priority": "Medium"
            })
        elif demographics["bmi"] > 25:
            recommendations.append({
                "category": "Lifestyle",
                "recommendation": "Maintain healthy weight through diet and exercise",
                "priority": "Low"
            })
        
        return RecommendationResponse(
             patient_id=patient_id,
             risk_score=prediction["risk_score"],
             risk_level=prediction["risk_category"],
             recommendations=recommendations,
             total_recommendations=len(recommendations),
             generated_at=datetime.now()
         )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation generation error for patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List available MLflow models"""
    try:
        # Mock model list
        models = [
            {
                "name": "cardiovascular_risk_model",
                "version": "1.0.0",
                "stage": "Production",
                "created_at": "2024-01-15T10:30:00Z",
                "accuracy": 0.85
            }
        ]
        return models
    except Exception as e:
        logger.error(f"Model listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model():
    """Trigger model retraining and refresh MLflow service"""
    try:
        # Mock model refresh
        return {
            "message": "Model refresh completed", 
            "timestamp": datetime.now(),
            "model_info": "mock_model_v1.0_refreshed"
        }
    except Exception as e:
        logger.error(f"Model refresh error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/monitoring/report", response_model=DriftReport)
async def get_drift_report():
    """Generate and return comprehensive drift monitoring report"""
    try:
        # Generate drift report using EvidentlyAI
        report = monitoring_service.generate_drift_report()
        
        return DriftReport(**report)
    
    except ValueError as e:
        logger.error(f"Drift report generation error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot generate drift report: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in drift report generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status", response_model=MonitoringStatus)
async def get_monitoring_status():
    """Get current monitoring system status"""
    try:
        latest_report = monitoring_service.get_latest_report()
        alerts = monitoring_service.get_alerts()
        thresholds = monitoring_service.get_thresholds()
        
        return MonitoringStatus(
            is_active=True,
            reference_data_loaded=monitoring_service._reference_data is not None,
            current_data_loaded=monitoring_service._current_data is not None,
            last_report_time=latest_report['timestamp'] if latest_report else None,
            total_alerts=len(alerts),
            thresholds=thresholds
        )
    
    except Exception as e:
        logger.error(f"Monitoring status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/alerts", response_model=AlertsResponse)
async def get_alerts(since: Optional[str] = None):
    """Get drift alerts, optionally filtered by timestamp"""
    try:
        since_datetime = None
        if since:
            try:
                since_datetime = datetime.fromisoformat(since)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid timestamp format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                )
        
        alerts = monitoring_service.get_alerts(since=since_datetime)
        
        return AlertsResponse(
            alerts=alerts,
            total_count=len(alerts),
            since=since
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alerts retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitoring/thresholds", response_model=MonitoringResponse)
async def update_drift_thresholds(thresholds: ThresholdUpdate):
    """Update drift detection thresholds"""
    try:
        # Convert to dict, filtering out None values
        threshold_updates = {
            k: v for k, v in thresholds.dict().items() 
            if v is not None
        }
        
        if not threshold_updates:
            raise HTTPException(
                status_code=400, 
                detail="At least one threshold value must be provided"
            )
        
        monitoring_service.update_thresholds(threshold_updates)
        
        return MonitoringResponse(
            success=True,
            message=f"Updated {len(threshold_updates)} threshold(s)",
            data={"updated_thresholds": threshold_updates}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Threshold update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/monitoring/alerts", response_model=MonitoringResponse)
async def clear_alerts():
    """Clear all stored alerts"""
    try:
        monitoring_service.clear_alerts()
        
        return MonitoringResponse(
            success=True,
            message="All alerts cleared successfully"
        )
    
    except Exception as e:
        logger.error(f"Alert clearing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients", response_model=PatientListResponse)
async def get_patients():
    """Get list of all patients for selection dropdown"""
    try:
        # Get patients from Supabase
        patients_data = await supabase_client.get_all_patients()
        
        # Transform data to match expected format
        patients = []
        for patient in patients_data:
            patient_formatted = {
                "id": patient['id'],
                "name": patient['name'],
                "age": patient['age'],
                "gender": patient['gender']
            }
            patients.append(patient_formatted)
        
        return PatientListResponse(
            patients=patients,
            total_count=len(patients),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error fetching patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient_data(patient_id: str):
    """Get detailed patient data including demographics, labs, and lifestyle"""
    try:
        # Get patient from Supabase
        patient_data = await supabase_client.get_patient_by_id(patient_id)
        
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Get related data
        lab_results = await supabase_client.get_patient_lab_results(patient_id)
        lifestyle_data = await supabase_client.get_patient_lifestyle_data(patient_id)
        
        # Format patient data
        patient = {
            "id": patient_data['id'],
            "name": patient_data['name'],
            "age": patient_data['age'],
            "gender": patient_data['gender'],
            "demographics": {
                "height": patient_data.get('height'),
                "weight": patient_data.get('weight'),
                "bmi": patient_data.get('bmi'),
                "blood_type": patient_data.get('blood_type'),
                "emergency_contact": patient_data.get('emergency_contact'),
                "phone": patient_data.get('phone')
            },
            "lab_results": {
                "cholesterol": lab_results[0].get('cholesterol') if lab_results else None,
                "hdl": lab_results[0].get('hdl') if lab_results else None,
                "ldl": lab_results[0].get('ldl') if lab_results else None,
                "triglycerides": lab_results[0].get('triglycerides') if lab_results else None,
                "glucose": lab_results[0].get('glucose') if lab_results else None,
                "hba1c": lab_results[0].get('hba1c') if lab_results else None,
                "systolic_bp": lab_results[0].get('systolic_bp') if lab_results else None,
                "diastolic_bp": lab_results[0].get('diastolic_bp') if lab_results else None,
                "last_test_date": lab_results[0].get('test_date') if lab_results else None
            },
            "lifestyle_data": {
                "smoking_status": lifestyle_data[0].get('smoking_status') if lifestyle_data else None,
                "alcohol_consumption": lifestyle_data[0].get('alcohol_consumption') if lifestyle_data else None,
                "exercise_frequency": lifestyle_data[0].get('exercise_frequency') if lifestyle_data else None,
                "diet_type": lifestyle_data[0].get('diet_type') if lifestyle_data else None,
                "stress_level": lifestyle_data[0].get('stress_level') if lifestyle_data else None,
                "sleep_hours": lifestyle_data[0].get('sleep_hours') if lifestyle_data else None,
                "family_history": lifestyle_data[0].get('family_history') if lifestyle_data else None
            },
            "created_at": patient_data.get('created_at', datetime.now()),
            "updated_at": patient_data.get('updated_at', datetime.now())
        }
        
        return PatientResponse(**patient)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/evaluation", response_model=ModelEvaluationResponse)
async def get_model_evaluation():
    """Get comprehensive model evaluation metrics for dashboard display"""
    try:
        # Generate synthetic test data for demonstration
        import pandas as pd
        import numpy as np
        
        # Create synthetic test dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X_test = pd.DataFrame({
            'age': np.random.normal(65, 15, n_samples),
            'bmi': np.random.normal(28, 5, n_samples),
            'systolic_bp': np.random.normal(140, 20, n_samples),
            'diastolic_bp': np.random.normal(85, 10, n_samples),
            'glucose': np.random.normal(120, 30, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'exercise_frequency': np.random.randint(0, 7, n_samples),
            'smoking_status': np.random.choice([0, 1], n_samples),
            'medication_adherence': np.random.uniform(0.5, 1.0, n_samples)
        })
        
        # Generate synthetic labels (30% positive cases)
        y_test = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        # Get model evaluation
        evaluation = await ml_model.get_model_evaluation(X_test, y_test)
        
        return ModelEvaluationResponse(**evaluation)
        
    except Exception as e:
        logger.error(f"Error getting model evaluation: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model evaluation")

@app.get("/cohort/risk-scores", response_model=CohortRiskResponse)
async def get_cohort_risk_scores():
    """Get risk scores for all patients in the cohort"""
    try:
        # Get all patients from Supabase
        patients_data = await supabase_client.get_all_patients()
        
        # Get existing predictions from Supabase
        all_predictions = await supabase_client.get_all_predictions()
        predictions_by_patient = {pred['patient_id']: pred for pred in all_predictions}
        
        # Calculate risk scores for each patient
        cohort_data = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        total_risk_score = 0
        
        for patient in patients_data:
            try:
                # Check if we have existing prediction
                existing_prediction = predictions_by_patient.get(patient['id'])
                
                if existing_prediction:
                    # Use existing prediction
                    patient_risk_data = {
                        "patient_id": patient["id"],
                        "name": patient["name"],
                        "age": patient["age"],
                        "gender": patient["gender"],
                        "risk_score": existing_prediction["risk_score"],
                        "risk_category": existing_prediction["risk_category"],
                        "last_updated": existing_prediction["prediction_date"]
                    }
                else:
                    # Generate new prediction
                    lab_results = await supabase_client.get_patient_lab_results(patient['id'])
                    lifestyle_data = await supabase_client.get_patient_lifestyle_data(patient['id'])
                    
                    patient_features = {
                        "age": patient["age"],
                        "bmi": patient.get("bmi", 25),
                        "systolic_bp": lab_results[0].get("systolic_bp", 120) if lab_results else 120,
                        "diastolic_bp": lab_results[0].get("diastolic_bp", 80) if lab_results else 80,
                        "glucose": lab_results[0].get("glucose", 100) if lab_results else 100,
                        "cholesterol": lab_results[0].get("cholesterol", 180) if lab_results else 180,
                        "exercise_frequency": lifestyle_data[0].get("exercise_frequency", 3) if lifestyle_data else 3,
                        "smoking_status": 1 if (lifestyle_data and lifestyle_data[0].get("smoking_status") == "Current") else 0,
                        "medication_adherence": 0.8  # Default adherence
                    }
                    
                    # Get risk prediction
                    prediction = await ml_model.predict(patient_features)
                    
                    # Store new prediction in Supabase
                    prediction_data = {
                        "patient_id": patient["id"],
                        "risk_score": prediction["risk_score"],
                        "risk_category": prediction["risk_category"],
                        "confidence": prediction["confidence"],
                        "model_version": prediction["model_version"],
                        "features_used": patient_features,
                        "prediction_date": datetime.now().isoformat()
                    }
                    
                    await supabase_client.create_prediction(prediction_data)
                    
                    patient_risk_data = {
                        "patient_id": patient["id"],
                        "name": patient["name"],
                        "age": patient["age"],
                        "gender": patient["gender"],
                        "risk_score": prediction["risk_score"],
                        "risk_category": prediction["risk_category"],
                        "last_updated": datetime.now().isoformat()
                    }
                
                cohort_data.append(patient_risk_data)
                risk_counts[patient_risk_data["risk_category"]] += 1
                total_risk_score += patient_risk_data["risk_score"]
                
            except Exception as e:
                logger.warning(f"Could not calculate risk for patient {patient['id']}: {e}")
                # Add patient with default risk
                patient_risk_data = {
                    "patient_id": patient["id"],
                    "name": patient["name"],
                    "age": patient["age"],
                    "gender": patient["gender"],
                    "risk_score": 0.3,  # Default medium-low risk
                    "risk_category": "Medium",
                    "last_updated": datetime.now().isoformat()
                }
                cohort_data.append(patient_risk_data)
                risk_counts["Medium"] += 1
                total_risk_score += 0.3
        
        # Sort by risk score (highest first)
        cohort_data.sort(key=lambda x: x["risk_score"], reverse=True)
        
        average_risk = total_risk_score / len(cohort_data) if cohort_data else 0
        
        return CohortRiskResponse(
            patients=cohort_data,
            total_patients=len(cohort_data),
            high_risk_count=risk_counts["High"],
            medium_risk_count=risk_counts["Medium"],
            low_risk_count=risk_counts["Low"],
            average_risk_score=round(average_risk, 3),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting cohort risk scores: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cohort risk scores")

@app.post("/explain/global")
async def get_global_explanations():
    """Get global model explanations using SHAP"""
    try:
        # Initialize explainer if not already done
        if explainer.shap_explainer is None:
            # Create synthetic training data for explainer initialization
            import pandas as pd
            import numpy as np
            
            # Generate sample data that matches our feature space
            np.random.seed(42)
            n_samples = 1000
            
            training_data = pd.DataFrame({
                'age': np.random.normal(65, 15, n_samples),
                'bmi': np.random.normal(28, 5, n_samples),
                'systolic_bp': np.random.normal(140, 20, n_samples),
                'diastolic_bp': np.random.normal(85, 10, n_samples),
                'glucose': np.random.normal(120, 30, n_samples),
                'cholesterol': np.random.normal(200, 40, n_samples),
                'exercise_frequency': np.random.randint(0, 7, n_samples),
                'smoking_status': np.random.binomial(1, 0.2, n_samples),
                'medication_adherence': np.random.beta(8, 2, n_samples)
            })
            
            # Initialize explainer with the ML model
            explainer.initialize_explainers(ml_model.model, training_data, mode='classification')
        
        # Generate global explanations using a representative sample
        sample_features = {
            'age': 70,
            'bmi': 30,
            'systolic_bp': 150,
            'diastolic_bp': 90,
            'glucose': 140,
            'cholesterol': 220,
            'exercise_frequency': 2,
            'smoking_status': 0,
            'medication_adherence': 0.7
        }
        
        explanations = await explainer.explain_prediction(sample_features)
        
        return {
            "global_explanations": explanations.get("shap_explanation", {}),
            "feature_importance": explanations.get("shap_explanation", {}).get("feature_contributions", []),
            "model_insights": {
                "most_important_factors": explanations.get("shap_explanation", {}).get("top_positive_features", []),
                "protective_factors": explanations.get("shap_explanation", {}).get("top_negative_features", []),
                "clinical_interpretation": "Global model explanations show the most influential factors across all patients for 90-day deterioration risk prediction."
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating global explanations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate global explanations: {str(e)}")

@app.post("/explain/local/{patient_id}")
async def get_local_explanations(patient_id: str):
    """Get local explanations for a specific patient"""
    try:
        # Get patient data from Supabase
        patient_data = await supabase_client.get_patient_by_id(patient_id)
        
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Get related data
        lab_results = await supabase_client.get_patient_lab_results(patient_id)
        lifestyle_data = await supabase_client.get_patient_lifestyle_data(patient_id)
        
        # Extract features for prediction
        patient_features = {
            "age": patient_data["age"],
            "bmi": patient_data.get("bmi", 25),
            "systolic_bp": lab_results[0].get("systolic_bp", 120) if lab_results else 120,
            "diastolic_bp": lab_results[0].get("diastolic_bp", 80) if lab_results else 80,
            "glucose": lab_results[0].get("glucose", 100) if lab_results else 100,
            "cholesterol": lab_results[0].get("cholesterol", 180) if lab_results else 180,
            "exercise_frequency": lifestyle_data[0].get("exercise_frequency", 3) if lifestyle_data else 3,
            "smoking_status": 1 if (lifestyle_data and lifestyle_data[0].get("smoking_status") == "Current") else 0,
            "medication_adherence": 0.8  # Default adherence
        }
        
        # Get prediction
        prediction = await ml_model.predict(patient_features)
        
        # Initialize explainer if needed
        if explainer.shap_explainer is None:
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            n_samples = 1000
            
            training_data = pd.DataFrame({
                'age': np.random.normal(65, 15, n_samples),
                'bmi': np.random.normal(28, 5, n_samples),
                'systolic_bp': np.random.normal(140, 20, n_samples),
                'diastolic_bp': np.random.normal(85, 10, n_samples),
                'glucose': np.random.normal(120, 30, n_samples),
                'cholesterol': np.random.normal(200, 40, n_samples),
                'exercise_frequency': np.random.randint(0, 7, n_samples),
                'smoking_status': np.random.binomial(1, 0.2, n_samples),
                'medication_adherence': np.random.beta(8, 2, n_samples)
            })
            
            explainer.initialize_explainers(ml_model.model, training_data, mode='classification')
        
        # Generate local explanations
        explanations = await explainer.explain_prediction(
            patient_features, 
            prediction=prediction["risk_score"],
            patient_id=patient_id
        )
        
        return {
            "patient_id": patient_id,
            "patient_name": patient_data["name"],
            "prediction": prediction,
            "local_explanations": explanations,
            "risk_drivers": explanations.get("shap_explanation", {}).get("top_positive_features", []),
            "protective_factors": explanations.get("shap_explanation", {}).get("top_negative_features", []),
            "clinical_recommendations": {
                "high_impact_factors": "Focus on the top risk drivers identified in the explanation",
                "actionable_insights": "Consider interventions for modifiable risk factors",
                "monitoring_priority": "High" if prediction["risk_score"] > 0.7 else "Medium" if prediction["risk_score"] > 0.4 else "Low"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating local explanations for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate local explanations: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
