from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Types of drift alerts"""
    DATA_DRIFT = "data_drift"
    TARGET_DRIFT = "target_drift"
    PREDICTION_DRIFT = "prediction_drift"
    DATA_QUALITY = "data_quality"

class HealthStatus(str, Enum):
    """Overall system health status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    ERROR = "error"

class RiskLevel(str, Enum):
    """Risk levels for monitoring"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class DriftedFeature(BaseModel):
    """Information about a feature that has drifted"""
    feature: str = Field(..., description="Name of the drifted feature")
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift score between 0 and 1")

class DataDriftInfo(BaseModel):
    """Data drift information"""
    dataset_drift: bool = Field(..., description="Whether dataset-level drift is detected")
    drift_share: float = Field(..., ge=0.0, le=1.0, description="Share of features showing drift")
    number_of_drifted_columns: int = Field(..., ge=0, description="Number of columns with drift")
    drifted_features: List[DriftedFeature] = Field(default=[], description="List of drifted features")

class TargetDriftInfo(BaseModel):
    """Target drift information"""
    drift_detected: bool = Field(..., description="Whether target drift is detected")
    drift_score: float = Field(..., ge=0.0, description="Target drift score")

class DataQualityInfo(BaseModel):
    """Data quality information"""
    missing_values_share: float = Field(..., ge=0.0, le=1.0, description="Share of missing values")
    duplicated_rows_share: float = Field(..., ge=0.0, le=1.0, description="Share of duplicated rows")
    constant_columns_share: float = Field(..., ge=0.0, le=1.0, description="Share of constant columns")

class PredictionDriftInfo(BaseModel):
    """Prediction drift information"""
    drift_detected: bool = Field(..., description="Whether prediction drift is detected")
    drift_score: float = Field(..., ge=0.0, description="Prediction drift score")

class Alert(BaseModel):
    """Drift alert information"""
    type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    threshold: float = Field(..., description="Threshold that was exceeded")
    actual_value: float = Field(..., description="Actual value that triggered the alert")
    timestamp: str = Field(..., description="Alert timestamp in ISO format")

class HealthSummary(BaseModel):
    """Overall system health summary"""
    overall_health: HealthStatus = Field(..., description="Overall system health status")
    risk_level: RiskLevel = Field(..., description="Current risk level")
    risk_factors: List[str] = Field(default=[], description="List of identified risk factors")
    recommendations: List[str] = Field(default=[], description="Recommendations for improvement")

class DriftReport(BaseModel):
    """Complete drift monitoring report"""
    timestamp: str = Field(..., description="Report generation timestamp in ISO format")
    data_drift: DataDriftInfo = Field(..., description="Data drift information")
    target_drift: TargetDriftInfo = Field(..., description="Target drift information")
    data_quality: DataQualityInfo = Field(..., description="Data quality information")
    prediction_drift: PredictionDriftInfo = Field(..., description="Prediction drift information")
    alerts: List[Alert] = Field(default=[], description="Current alerts")
    summary: HealthSummary = Field(..., description="Overall health summary")
    error: Optional[str] = Field(None, description="Error message if report generation failed")

class MonitoringStatus(BaseModel):
    """Current monitoring system status"""
    is_active: bool = Field(..., description="Whether monitoring is active")
    reference_data_loaded: bool = Field(..., description="Whether reference data is loaded")
    current_data_loaded: bool = Field(..., description="Whether current data is loaded")
    last_report_time: Optional[str] = Field(None, description="Last report generation time")
    total_alerts: int = Field(..., ge=0, description="Total number of alerts")
    thresholds: Dict[str, float] = Field(..., description="Current drift thresholds")

class ThresholdUpdate(BaseModel):
    """Request model for updating drift thresholds"""
    data_drift: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data drift threshold")
    target_drift: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target drift threshold")
    prediction_drift: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction drift threshold")
    missing_values: Optional[float] = Field(None, ge=0.0, le=1.0, description="Missing values threshold")

class AlertsResponse(BaseModel):
    """Response model for alerts endpoint"""
    alerts: List[Alert] = Field(..., description="List of alerts")
    total_count: int = Field(..., ge=0, description="Total number of alerts")
    since: Optional[str] = Field(None, description="Timestamp filter applied")

class MonitoringResponse(BaseModel):
    """Generic monitoring response"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")

class DataUpload(BaseModel):
    """Model for data upload requests"""
    data_type: str = Field(..., description="Type of data (reference or current)")
    file_path: Optional[str] = Field(None, description="Path to data file")
    data_format: str = Field(default="csv", description="Data format (csv, json, parquet)")
    target_column: Optional[str] = Field(None, description="Name of target column")

class ReportRequest(BaseModel):
    """Request model for generating drift reports"""
    include_raw_data: bool = Field(default=False, description="Whether to include raw EvidentlyAI report")
    alert_threshold_override: Optional[Dict[str, float]] = Field(None, description="Temporary threshold overrides")