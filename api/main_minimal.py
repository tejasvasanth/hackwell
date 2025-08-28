from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from ml.monitoring_service import monitoring_service
from api.models.monitoring import (
    DriftReport, MonitoringStatus, AlertsResponse, 
    MonitoringResponse, ThresholdUpdate
)

app = FastAPI(
    title="ML Pipeline API - Monitoring Only",
    description="FastAPI application for ML monitoring endpoints",
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

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Pipeline API - Monitoring Only",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "monitoring_service": "available"
    }

# Monitoring endpoints
@app.get("/monitoring/report")
async def get_drift_report():
    """Generate and return drift analysis report"""
    try:
        report = monitoring_service.generate_drift_report()
        return report
    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get current monitoring system status"""
    try:
        status = monitoring_service.get_monitoring_status()
        return status
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/alerts")
async def get_alerts():
    """Get current alerts"""
    try:
        alerts = monitoring_service.get_alerts()
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitoring/thresholds")
async def update_thresholds(threshold_update: ThresholdUpdate):
    """Update drift detection thresholds"""
    try:
        # Create thresholds dict from the request
        thresholds = {}
        if threshold_update.data_drift is not None:
            thresholds['data_drift'] = threshold_update.data_drift
        if threshold_update.target_drift is not None:
            thresholds['target_drift'] = threshold_update.target_drift
        if threshold_update.prediction_drift is not None:
            thresholds['prediction_drift'] = threshold_update.prediction_drift
        if threshold_update.missing_values is not None:
            thresholds['missing_values'] = threshold_update.missing_values
        
        monitoring_service.update_thresholds(thresholds)
        
        return {
            "message": "Thresholds updated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/monitoring/alerts")
async def clear_alerts():
    """Clear all alerts"""
    try:
        monitoring_service.clear_alerts()
        
        return {
            "message": "All alerts cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)