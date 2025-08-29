import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import threading
import json
from pathlib import Path
from loguru import logger
import sys
import os

from evidently import Report
from evidently import DataDefinition
from evidently.metrics import (
    DriftedColumnsCount,
    DatasetMissingValueCount,
    ValueDrift,
    MissingValueCount
)

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client

class MonitoringService:
    """Service for monitoring model drift, data drift, and label drift using EvidentlyAI"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._reference_data: Optional[pd.DataFrame] = None
        self._current_data: Optional[pd.DataFrame] = None
        self._predictions_data: Optional[pd.DataFrame] = None
        self._drift_thresholds = {
            'data_drift': 0.5,  # 50% of features showing drift
            'target_drift': 0.1,  # 10% drift in target distribution
            'prediction_drift': 0.1,  # 10% drift in prediction distribution
            'missing_values': 0.05  # 5% missing values threshold
        }
        self._alerts: List[Dict[str, Any]] = []
        self._reports_cache: Dict[str, Any] = {}
        
        # Data definition for healthcare data
        self.data_definition = DataDefinition(
            numerical_columns=['age', 'cholesterol', 'blood_pressure', 'bmi', 'exercise_frequency', 'risk_score', 'prediction'],
            categorical_columns=['diabetes', 'smoking', 'family_history', 'gender', 'target']
        )
        
        logger.info("MonitoringService initialized")
        
        # Initialize Supabase client
        try:
            self.supabase = get_supabase_client()
            logger.info("Supabase client initialized for monitoring")
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase client: {e}")
            self.supabase = None
    
    def set_reference_data(self, data: pd.DataFrame, target_column: str = 'target'):
        """Set reference dataset for drift comparison"""
        with self._lock:
            self._reference_data = data.copy()
            # Update data definition if needed
            if target_column in data.columns and target_column != 'target':
                if target_column not in self.data_definition.categorical_columns:
                    self.data_definition.categorical_columns.append(target_column)
            logger.info(f"Reference data set with {len(data)} rows")
    
    def add_current_data(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None):
        """Add current data for drift monitoring"""
        with self._lock:
            # Add predictions if provided
            if predictions is not None:
                data = data.copy()
                data['prediction'] = predictions
            
            self._current_data = data
            logger.info(f"Current data updated with {len(data)} rows")
    
    def generate_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report using EvidentlyAI"""
        with self._lock:
            if self._reference_data is None or self._current_data is None:
                raise ValueError("Reference and current data must be set before generating report")
            
            try:
                # Create simplified report with available metrics
                metrics = [
                    DriftedColumnsCount(),
                    DatasetMissingValueCount()
                ]
                
                # Add ValueDrift metrics only for columns present in both datasets
                if 'target' in self._current_data.columns and 'target' in self._reference_data.columns:
                    metrics.append(ValueDrift(column='target'))
                
                if 'prediction' in self._current_data.columns and 'prediction' in self._reference_data.columns:
                    metrics.append(ValueDrift(column='prediction'))
                
                report = Report(metrics=metrics)
                
                # Run the report
                report.run(
                    reference_data=self._reference_data,
                    current_data=self._current_data
                )
                
                # Extract report data
                report_dict = {'metrics': [metric.dict() for metric in report.metrics]}
                
                # Process and structure the results
                drift_summary = self._process_drift_results(report_dict)
                
                # Check for alerts
                self._check_drift_alerts(drift_summary)
                
                # Cache the report
                self._reports_cache['latest'] = {
                    'report': drift_summary,
                    'timestamp': datetime.now(),
                    'raw_report': report_dict
                }
                
                # Store report in Supabase
                self.store_monitoring_report(drift_summary)
                
                logger.info("Drift report generated successfully")
                return drift_summary
                
            except Exception as e:
                logger.error(f"Error generating drift report: {str(e)}")
                # Return a basic error report
                return self._create_error_report(str(e))
    
    def _create_error_report(self, error_message: str) -> Dict[str, Any]:
        """Create a basic error report when EvidentlyAI fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_drift': {
                'dataset_drift': False,
                'drift_share': 0.0,
                'number_of_drifted_columns': 0,
                'drifted_features': []
            },
            'target_drift': {
                'drift_detected': False,
                'drift_score': 0.0
            },
            'data_quality': {
                'missing_values_share': 0.0,
                'duplicated_rows_share': 0.0,
                'constant_columns_share': 0.0
            },
            'prediction_drift': {
                'drift_detected': False,
                'drift_score': 0.0
            },
            'alerts': [],
            'summary': {
                'overall_health': 'error',
                'risk_level': 'unknown',
                'risk_factors': ['monitoring_error'],
                'recommendations': [f"Fix monitoring error: {error_message}"]
            },
            'error': error_message
        }
    
    def _process_drift_results(self, report_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process EvidentlyAI report results into structured format"""
        try:
            metrics = report_dict.get('metrics', [])
            
            # Initialize result structure
            result = {
                'timestamp': datetime.now().isoformat(),
                'data_drift': {
                    'dataset_drift': False,
                    'drift_share': 0.0,
                    'number_of_drifted_columns': 0,
                    'drifted_features': []
                },
                'target_drift': {
                    'drift_detected': False,
                    'drift_score': 0.0
                },
                'data_quality': {
                    'missing_values_share': 0.0,
                    'duplicated_rows_share': 0.0,
                    'constant_columns_share': 0.0
                },
                'prediction_drift': {
                    'drift_detected': False,
                    'drift_score': 0.0
                },
                'alerts': [],
                'summary': {
                    'overall_health': 'good',
                    'risk_level': 'low'
                }
            }
            
            # Process each metric
            for metric in metrics:
                metric_type = metric.get('metric', '')
                metric_result = metric.get('result', {})
                
                if 'DriftedColumnsCount' in metric_type:
                    drifted_count = metric_result.get('current', 0)
                    total_columns = len(self._current_data.columns)
                    drift_share = drifted_count / total_columns if total_columns > 0 else 0
                    
                    result['data_drift']['number_of_drifted_columns'] = drifted_count
                    result['data_drift']['drift_share'] = drift_share
                    result['data_drift']['dataset_drift'] = drift_share > self._drift_thresholds['data_drift']
                
                elif 'DatasetMissingValueCount' in metric_type:
                    missing_count = metric_result.get('current', 0)
                    total_values = len(self._current_data) * len(self._current_data.columns)
                    missing_share = missing_count / total_values if total_values > 0 else 0
                    
                    result['data_quality']['missing_values_share'] = missing_share
                
                elif 'ValueDrift' in metric_type:
                    column_name = metric.get('column', '')
                    drift_detected = metric_result.get('drift_detected', False)
                    drift_score = metric_result.get('drift_score', 0.0)
                    
                    if column_name == 'target':
                        result['target_drift']['drift_detected'] = drift_detected
                        result['target_drift']['drift_score'] = drift_score
                    elif column_name == 'prediction':
                        result['prediction_drift']['drift_detected'] = drift_detected
                        result['prediction_drift']['drift_score'] = drift_score
            
            # Calculate overall health
            result['summary'] = self._calculate_overall_health(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing drift results: {str(e)}")
            return self._create_error_report(f"Processing error: {str(e)}")
    
    def _calculate_overall_health(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health based on drift metrics"""
        risk_factors = []
        
        # Check data drift
        if result['data_drift']['dataset_drift']:
            risk_factors.append('data_drift')
        
        # Check target drift
        if result['target_drift']['drift_detected']:
            risk_factors.append('target_drift')
        
        # Check prediction drift
        if result['prediction_drift']['drift_detected']:
            risk_factors.append('prediction_drift')
        
        # Check data quality
        if result['data_quality']['missing_values_share'] > self._drift_thresholds['missing_values']:
            risk_factors.append('data_quality')
        
        # Determine overall health
        if len(risk_factors) == 0:
            health = 'excellent'
            risk_level = 'low'
        elif len(risk_factors) == 1:
            health = 'good'
            risk_level = 'medium'
        elif len(risk_factors) == 2:
            health = 'fair'
            risk_level = 'high'
        else:
            health = 'poor'
            risk_level = 'critical'
        
        return {
            'overall_health': health,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._get_health_recommendations(risk_factors)
        }
    
    def _get_health_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Get recommendations based on detected risk factors"""
        recommendations = []
        
        if 'data_drift' in risk_factors:
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Investigate changes in data collection process")
        
        if 'target_drift' in risk_factors:
            recommendations.append("Review target variable definition and collection")
            recommendations.append("Consider updating model with new target distribution")
        
        if 'prediction_drift' in risk_factors:
            recommendations.append("Monitor model performance closely")
            recommendations.append("Consider model recalibration")
        
        if 'data_quality' in risk_factors:
            recommendations.append("Improve data validation and cleaning processes")
            recommendations.append("Investigate data pipeline for quality issues")
        
        return recommendations
    
    def _check_drift_alerts(self, drift_summary: Dict[str, Any]):
        """Check drift thresholds and generate alerts"""
        alerts = []
        
        # Check data drift threshold
        if drift_summary['data_drift']['drift_share'] > self._drift_thresholds['data_drift']:
            alerts.append({
                'type': 'data_drift',
                'severity': 'high',
                'message': f"Data drift detected in {drift_summary['data_drift']['drift_share']:.2%} of features",
                'threshold': self._drift_thresholds['data_drift'],
                'actual_value': drift_summary['data_drift']['drift_share'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Check target drift threshold
        if drift_summary['target_drift']['drift_score'] > self._drift_thresholds['target_drift']:
            alerts.append({
                'type': 'target_drift',
                'severity': 'critical',
                'message': f"Target drift detected with score {drift_summary['target_drift']['drift_score']:.3f}",
                'threshold': self._drift_thresholds['target_drift'],
                'actual_value': drift_summary['target_drift']['drift_score'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Check prediction drift threshold
        if drift_summary['prediction_drift']['drift_score'] > self._drift_thresholds['prediction_drift']:
            alerts.append({
                'type': 'prediction_drift',
                'severity': 'high',
                'message': f"Prediction drift detected with score {drift_summary['prediction_drift']['drift_score']:.3f}",
                'threshold': self._drift_thresholds['prediction_drift'],
                'actual_value': drift_summary['prediction_drift']['drift_score'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Check data quality threshold
        if drift_summary['data_quality']['missing_values_share'] > self._drift_thresholds['missing_values']:
            alerts.append({
                'type': 'data_quality',
                'severity': 'medium',
                'message': f"High missing values detected: {drift_summary['data_quality']['missing_values_share']:.2%}",
                'threshold': self._drift_thresholds['missing_values'],
                'actual_value': drift_summary['data_quality']['missing_values_share'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Store alerts
        self._alerts.extend(alerts)
        
        # Log alerts and store in Supabase
        for alert in alerts:
            logger.warning(f"DRIFT ALERT [{alert['severity'].upper()}]: {alert['message']}")
            self._store_alert_in_supabase(alert)
    
    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest drift report"""
        with self._lock:
            return self._reports_cache.get('latest')
    
    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alerts since specified time"""
        with self._lock:
            if since is None:
                return self._alerts.copy()
            
            return [
                alert for alert in self._alerts
                if datetime.fromisoformat(alert['timestamp']) >= since
            ]
    
    def clear_alerts(self):
        """Clear all stored alerts"""
        with self._lock:
            self._alerts.clear()
            logger.info("All alerts cleared")
    
    def update_thresholds(self, thresholds: Dict[str, float]):
        """Update drift detection thresholds"""
        with self._lock:
            self._drift_thresholds.update(thresholds)
            logger.info(f"Drift thresholds updated: {thresholds}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current drift detection thresholds"""
        with self._lock:
            return self._drift_thresholds.copy()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        with self._lock:
            # Get latest report if available
            latest_report = self._reports_cache.get('latest')
            
            # Count active alerts by severity
            alert_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            for alert in self._alerts:
                severity = alert.get('severity', 'low')
                if severity in alert_counts:
                    alert_counts[severity] += 1
            
            # Determine overall health status
            if alert_counts['critical'] > 0:
                health_status = 'critical'
            elif alert_counts['high'] > 0:
                health_status = 'degraded'
            elif alert_counts['medium'] > 0:
                health_status = 'warning'
            else:
                health_status = 'healthy'
            
            # Calculate uptime (simplified)
            uptime_hours = 24  # Placeholder
            
            return {
                'health_status': health_status,
                'last_check': latest_report['timestamp'] if latest_report else None,
                'active_alerts': len(self._alerts),
                'alert_breakdown': alert_counts,
                'data_sources': {
                    'reference_data': self._reference_data is not None,
                    'current_data': self._current_data is not None,
                    'reference_rows': len(self._reference_data) if self._reference_data is not None else 0,
                    'current_rows': len(self._current_data) if self._current_data is not None else 0
                },
                'thresholds': self._drift_thresholds.copy(),
                'uptime_hours': uptime_hours,
                'timestamp': datetime.now().isoformat()
            }

    def _store_alert_in_supabase(self, alert: Dict[str, Any]):
        """Store alert in Supabase database"""
        if not self.supabase:
            return
        
        try:
            # Convert alert to Supabase format
            supabase_alert = {
                'alert_type': alert['type'],
                'severity': alert['severity'],
                'title': f"{alert['type'].replace('_', ' ').title()} Alert",
                'description': alert['message'],
                'status': 'active',
                'triggered_at': alert['timestamp'],
                'metadata': json.dumps(alert.get('metadata', {})) if alert.get('metadata') else None
            }
            
            self.supabase.client.table('alerts').insert(supabase_alert).execute()
            logger.debug(f"Alert stored in Supabase: {alert['type']}")
            
        except Exception as e:
            logger.error(f"Failed to store alert in Supabase: {e}")
    
    def store_monitoring_report(self, report: Dict[str, Any]):
        """Store monitoring report in Supabase"""
        if not self.supabase:
            return
        
        try:
            # Store in monitoring_reports table
            monitoring_report = {
                'report_type': 'drift_analysis',
                'timestamp': report.get('timestamp', datetime.now().isoformat()),
                'data_drift_detected': report.get('data_drift', {}).get('dataset_drift', False),
                'target_drift_detected': report.get('target_drift', {}).get('drift_detected', False),
                'prediction_drift_detected': report.get('prediction_drift', {}).get('drift_detected', False),
                'data_quality_score': 1.0 - report.get('data_quality', {}).get('missing_values_share', 0.0),
                'overall_health': report.get('summary', {}).get('overall_health', 'unknown'),
                'risk_level': report.get('summary', {}).get('risk_level', 'unknown'),
                'report_data': json.dumps(report)
            }
            
            self.supabase.client.table('monitoring_reports').insert(monitoring_report).execute()
            logger.debug("Monitoring report stored in Supabase")
            
        except Exception as e:
            logger.error(f"Failed to store monitoring report in Supabase: {e}")
    
    def load_data_from_supabase(self, limit: int = 1000) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load patient data from Supabase for monitoring"""
        if not self.supabase:
            logger.warning("Supabase client not available")
            return None, None
        
        try:
            # Get patient data with lab results
            patients_response = self.supabase.client.table('patients').select('*').limit(limit).execute()
            lab_results_response = self.supabase.client.table('lab_results').select('*').limit(limit * 5).execute()
            
            if not patients_response.data or not lab_results_response.data:
                logger.warning("No data found in Supabase")
                return None, None
            
            # Convert to DataFrames
            patients_df = pd.DataFrame(patients_response.data)
            lab_results_df = pd.DataFrame(lab_results_response.data)
            
            # Pivot lab results to get features per patient
            lab_pivot = lab_results_df.pivot_table(
                index='patient_id',
                columns='test_name',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Merge patient data with lab results
            monitoring_data = patients_df.merge(lab_pivot, left_on='id', right_on='patient_id', how='left')
            
            # Add required columns for monitoring
            monitoring_data['diabetes'] = (monitoring_data.get('blood_glucose', 100) > 126).astype(int)
            monitoring_data['smoking'] = np.random.choice([0, 1], len(monitoring_data), p=[0.7, 0.3])  # Placeholder
            monitoring_data['family_history'] = np.random.choice([0, 1], len(monitoring_data), p=[0.6, 0.4])  # Placeholder
            monitoring_data['gender'] = (monitoring_data['gender'] == 'Male').astype(int)
            monitoring_data['exercise_frequency'] = np.random.poisson(3, len(monitoring_data))  # Placeholder
            
            # Calculate risk score
            monitoring_data['risk_score'] = (
                monitoring_data['age'] * 0.02 +
                monitoring_data.get('cholesterol', 200) * 0.01 +
                monitoring_data.get('blood_pressure_systolic', 120) * 0.015 +
                monitoring_data['bmi'] * 0.1 +
                monitoring_data['diabetes'] * 10 +
                monitoring_data['smoking'] * 8 +
                monitoring_data['family_history'] * 5
            )
            
            monitoring_data['target'] = (monitoring_data['risk_score'] > monitoring_data['risk_score'].median()).astype(int)
            monitoring_data['prediction'] = np.random.uniform(0, 1, len(monitoring_data))  # Placeholder
            
            # Split into reference and current data (70/30 split)
            split_idx = int(len(monitoring_data) * 0.7)
            reference_data = monitoring_data.iloc[:split_idx].copy()
            current_data = monitoring_data.iloc[split_idx:].copy()
            
            logger.info(f"Loaded {len(reference_data)} reference samples and {len(current_data)} current samples from Supabase")
            return reference_data, current_data
            
        except Exception as e:
            logger.error(f"Failed to load data from Supabase: {e}")
            return None, None

# Global monitoring service instance
monitoring_service = MonitoringService()