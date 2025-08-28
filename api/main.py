from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime, timedelta
from loguru import logger

from config.settings import settings
# from ml.mlflow_service import mlflow_service
# from ml.explainer import ModelExplainer
# from ml.metrics_service import metrics_service
# from ml.recommendation_service import recommendation_service
from ml.monitoring_service import monitoring_service
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
# explainer = ModelExplainer()

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
        # Mock patient data - in production, this would come from database
        patients = [
            {"id": "P001", "name": "John Smith", "age": 65, "gender": "Male"},
            {"id": "P002", "name": "Sarah Johnson", "age": 58, "gender": "Female"},
            {"id": "P003", "name": "Michael Brown", "age": 72, "gender": "Male"},
            {"id": "P004", "name": "Emily Davis", "age": 45, "gender": "Female"},
            {"id": "P005", "name": "Robert Wilson", "age": 68, "gender": "Male"}
        ]
        
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
        # Mock patient data - in production, this would come from database
        patient_data = {
            "P001": {
                "id": "P001",
                "name": "John Smith",
                "age": 65,
                "gender": "Male",
                "demographics": {
                    "height": 175,
                    "weight": 80,
                    "bmi": 26.1,
                    "blood_type": "O+",
                    "emergency_contact": "Jane Smith (Wife)",
                    "phone": "+1-555-0123"
                },
                "lab_results": {
                    "cholesterol": 240,
                    "hdl": 45,
                    "ldl": 160,
                    "triglycerides": 180,
                    "glucose": 110,
                    "hba1c": 6.2,
                    "systolic_bp": 145,
                    "diastolic_bp": 90,
                    "last_test_date": "2024-01-15"
                },
                "lifestyle_data": {
                    "smoking_status": "Former",
                    "alcohol_consumption": "Moderate",
                    "exercise_frequency": 3,
                    "diet_type": "Mediterranean",
                    "stress_level": "Medium",
                    "sleep_hours": 7,
                    "family_history": True
                }
            },
            "P002": {
                "id": "P002",
                "name": "Sarah Johnson",
                "age": 58,
                "gender": "Female",
                "demographics": {
                    "height": 165,
                    "weight": 68,
                    "bmi": 25.0,
                    "blood_type": "A+",
                    "emergency_contact": "Tom Johnson (Husband)",
                    "phone": "+1-555-0456"
                },
                "lab_results": {
                    "cholesterol": 200,
                    "hdl": 55,
                    "ldl": 120,
                    "triglycerides": 125,
                    "glucose": 95,
                    "hba1c": 5.8,
                    "systolic_bp": 125,
                    "diastolic_bp": 80,
                    "last_test_date": "2024-01-20"
                },
                "lifestyle_data": {
                    "smoking_status": "Never",
                    "alcohol_consumption": "Light",
                    "exercise_frequency": 5,
                    "diet_type": "Vegetarian",
                    "stress_level": "Low",
                    "sleep_hours": 8,
                    "family_history": False
                }
            },
            "P003": {
                "id": "P003",
                "name": "Michael Brown",
                "age": 72,
                "gender": "Male",
                "demographics": {
                    "height": 180,
                    "weight": 85,
                    "bmi": 26.2,
                    "blood_type": "B+",
                    "emergency_contact": "Lisa Brown (Daughter)",
                    "phone": "+1-555-0789"
                },
                "lab_results": {
                    "cholesterol": 280,
                    "hdl": 40,
                    "ldl": 190,
                    "triglycerides": 250,
                    "glucose": 130,
                    "hba1c": 6.8,
                    "systolic_bp": 160,
                    "diastolic_bp": 95,
                    "last_test_date": "2024-01-10"
                },
                "lifestyle_data": {
                    "smoking_status": "Current",
                    "alcohol_consumption": "Heavy",
                    "exercise_frequency": 1,
                    "diet_type": "Standard",
                    "stress_level": "High",
                    "sleep_hours": 6,
                    "family_history": True
                }
            },
            "P004": {
                "id": "P004",
                "name": "Emily Davis",
                "age": 45,
                "gender": "Female",
                "demographics": {
                    "height": 160,
                    "weight": 55,
                    "bmi": 21.5,
                    "blood_type": "AB+",
                    "emergency_contact": "David Davis (Husband)",
                    "phone": "+1-555-0321"
                },
                "lab_results": {
                    "cholesterol": 180,
                    "hdl": 65,
                    "ldl": 100,
                    "triglycerides": 75,
                    "glucose": 85,
                    "hba1c": 5.2,
                    "systolic_bp": 110,
                    "diastolic_bp": 70,
                    "last_test_date": "2024-01-25"
                },
                "lifestyle_data": {
                    "smoking_status": "Never",
                    "alcohol_consumption": "None",
                    "exercise_frequency": 6,
                    "diet_type": "Vegan",
                    "stress_level": "Low",
                    "sleep_hours": 8,
                    "family_history": False
                }
            },
            "P005": {
                "id": "P005",
                "name": "Robert Wilson",
                "age": 68,
                "gender": "Male",
                "demographics": {
                    "height": 170,
                    "weight": 90,
                    "bmi": 31.1,
                    "blood_type": "O-",
                    "emergency_contact": "Mary Wilson (Wife)",
                    "phone": "+1-555-0654"
                },
                "lab_results": {
                    "cholesterol": 260,
                    "hdl": 35,
                    "ldl": 180,
                    "triglycerides": 225,
                    "glucose": 140,
                    "hba1c": 7.1,
                    "systolic_bp": 150,
                    "diastolic_bp": 92,
                    "last_test_date": "2024-01-12"
                },
                "lifestyle_data": {
                    "smoking_status": "Former",
                    "alcohol_consumption": "Moderate",
                    "exercise_frequency": 2,
                    "diet_type": "Low-carb",
                    "stress_level": "Medium",
                    "sleep_hours": 6,
                    "family_history": True
                }
            }
        }
        
        if patient_id not in patient_data:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        patient = patient_data[patient_id]
        patient["created_at"] = datetime.now() - timedelta(days=30)
        patient["updated_at"] = datetime.now()
        
        return PatientResponse(**patient)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
