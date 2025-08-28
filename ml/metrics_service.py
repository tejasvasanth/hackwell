import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from scipy import stats
import asyncio
from collections import deque
import json

from config.supabase_client import supabase

class MetricsService:
    """Service for tracking model performance metrics and drift detection"""
    
    def __init__(self, max_predictions_cache: int = 10000):
        self.predictions_cache = deque(maxlen=max_predictions_cache)
        self.ground_truth_cache = deque(maxlen=max_predictions_cache)
        self.feature_cache = deque(maxlen=max_predictions_cache)
        
        # Drift detection settings
        self.drift_threshold = 0.05  # p-value threshold for statistical tests
        self.min_samples_for_drift = 100
        self.reference_data = None
        
        # Performance tracking
        self.performance_history = []
        self.alert_thresholds = {
            "accuracy_drop": 0.05,  # Alert if accuracy drops by 5%
            "auc_drop": 0.05,
            "drift_detected": True
        }
    
    async def log_prediction(self, 
                           patient_id: str, 
                           features: Dict[str, Any], 
                           prediction: float, 
                           probability: float,
                           model_version: str = None,
                           ground_truth: float = None) -> None:
        """Log a prediction for metrics tracking"""
        try:
            # Add to cache
            prediction_record = {
                "patient_id": patient_id,
                "timestamp": datetime.now(),
                "prediction": prediction,
                "probability": probability,
                "model_version": model_version,
                "features": features
            }
            
            self.predictions_cache.append(prediction_record)
            self.feature_cache.append(features)
            
            if ground_truth is not None:
                self.ground_truth_cache.append({
                    "patient_id": patient_id,
                    "timestamp": datetime.now(),
                    "ground_truth": ground_truth
                })
            
            # Store in database
            await self._store_prediction_db(prediction_record, ground_truth)
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    async def _store_prediction_db(self, prediction_record: Dict[str, Any], ground_truth: float = None):
        """Store prediction in Supabase database"""
        try:
            data = {
                "patient_id": prediction_record["patient_id"],
                "timestamp": prediction_record["timestamp"].isoformat(),
                "prediction": prediction_record["prediction"],
                "probability": prediction_record["probability"],
                "model_version": prediction_record["model_version"],
                "features": json.dumps(prediction_record["features"]),
                "ground_truth": ground_truth
            }
            
            result = supabase.table("model_predictions").insert(data).execute()
            
        except Exception as e:
            logger.error(f"Error storing prediction in database: {e}")
    
    async def get_evaluation_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """Calculate evaluation metrics for recent predictions"""
        try:
            # Get recent predictions with ground truth
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query from database
            result = supabase.table("model_predictions").select("*").gte(
                "timestamp", cutoff_date.isoformat()
            ).not_.is_("ground_truth", "null").execute()
            
            if not result.data:
                return {
                    "error": "No ground truth data available for evaluation",
                    "period_days": days_back,
                    "sample_count": 0
                }
            
            # Extract predictions and ground truth
            predictions = []
            probabilities = []
            ground_truths = []
            
            for record in result.data:
                predictions.append(record["prediction"])
                probabilities.append(record["probability"])
                ground_truths.append(record["ground_truth"])
            
            # Calculate confusion matrix
            cm = confusion_matrix(ground_truths, predictions)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(predictions))
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate AUC if we have probabilities
            auc_score = None
            if len(set(ground_truths)) > 1:  # Need both classes for AUC
                try:
                    auc_score = roc_auc_score(ground_truths, probabilities)
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")
            
            metrics = {
                "period_days": days_back,
                "sample_count": len(predictions),
                "confusion_matrix": {
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn)
                },
                "performance_metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "f1_score": float(f1_score),
                    "auc_score": float(auc_score) if auc_score is not None else None
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for performance alerts
            alerts = await self._check_performance_alerts(metrics)
            metrics["alerts"] = alerts
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {e}")
            return {"error": str(e)}
    
    async def _check_performance_alerts(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance degradation alerts"""
        alerts = []
        
        try:
            # Get historical performance for comparison
            if len(self.performance_history) > 0:
                last_metrics = self.performance_history[-1]
                
                current_acc = current_metrics["performance_metrics"]["accuracy"]
                last_acc = last_metrics["performance_metrics"]["accuracy"]
                
                if current_acc < last_acc - self.alert_thresholds["accuracy_drop"]:
                    alerts.append({
                        "type": "accuracy_drop",
                        "severity": "warning",
                        "message": f"Accuracy dropped from {last_acc:.3f} to {current_acc:.3f}",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check AUC if available
                current_auc = current_metrics["performance_metrics"]["auc_score"]
                last_auc = last_metrics["performance_metrics"]["auc_score"]
                
                if current_auc and last_auc and current_auc < last_auc - self.alert_thresholds["auc_drop"]:
                    alerts.append({
                        "type": "auc_drop",
                        "severity": "warning",
                        "message": f"AUC dropped from {last_auc:.3f} to {current_auc:.3f}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Store current metrics in history
            self.performance_history.append(current_metrics)
            
            # Keep only last 30 performance records
            if len(self.performance_history) > 30:
                self.performance_history.pop(0)
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
        
        return alerts
    
    async def detect_drift(self, reference_period_days: int = 30) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        try:
            # Get reference data (older period)
            reference_end = datetime.now() - timedelta(days=reference_period_days)
            reference_start = reference_end - timedelta(days=reference_period_days)
            
            # Get current data (recent period)
            current_start = datetime.now() - timedelta(days=7)  # Last week
            
            # Query reference data
            ref_result = supabase.table("model_predictions").select("features").gte(
                "timestamp", reference_start.isoformat()
            ).lte("timestamp", reference_end.isoformat()).execute()
            
            # Query current data
            curr_result = supabase.table("model_predictions").select("features").gte(
                "timestamp", current_start.isoformat()
            ).execute()
            
            if len(ref_result.data) < self.min_samples_for_drift or len(curr_result.data) < self.min_samples_for_drift:
                return {
                    "drift_detected": False,
                    "reason": "Insufficient data for drift detection",
                    "reference_samples": len(ref_result.data),
                    "current_samples": len(curr_result.data),
                    "min_required": self.min_samples_for_drift
                }
            
            # Extract features
            ref_features = [json.loads(record["features"]) for record in ref_result.data]
            curr_features = [json.loads(record["features"]) for record in curr_result.data]
            
            # Convert to DataFrames
            ref_df = pd.DataFrame(ref_features)
            curr_df = pd.DataFrame(curr_features)
            
            # Perform drift tests for each feature
            drift_results = {}
            overall_drift = False
            
            for feature in ref_df.columns:
                if feature in curr_df.columns:
                    ref_values = ref_df[feature].dropna()
                    curr_values = curr_df[feature].dropna()
                    
                    if len(ref_values) > 10 and len(curr_values) > 10:
                        # Kolmogorov-Smirnov test for distribution drift
                        ks_stat, ks_p_value = stats.ks_2samp(ref_values, curr_values)
                        
                        # Mann-Whitney U test for median shift
                        mw_stat, mw_p_value = stats.mannwhitneyu(ref_values, curr_values, alternative='two-sided')
                        
                        feature_drift = ks_p_value < self.drift_threshold or mw_p_value < self.drift_threshold
                        
                        drift_results[feature] = {
                            "drift_detected": feature_drift,
                            "ks_statistic": float(ks_stat),
                            "ks_p_value": float(ks_p_value),
                            "mw_statistic": float(mw_stat),
                            "mw_p_value": float(mw_p_value),
                            "reference_mean": float(ref_values.mean()),
                            "current_mean": float(curr_values.mean()),
                            "reference_std": float(ref_values.std()),
                            "current_std": float(curr_values.std())
                        }
                        
                        if feature_drift:
                            overall_drift = True
            
            # Create drift alert if detected
            alerts = []
            if overall_drift:
                drifted_features = [f for f, r in drift_results.items() if r["drift_detected"]]
                alerts.append({
                    "type": "data_drift",
                    "severity": "warning",
                    "message": f"Data drift detected in features: {', '.join(drifted_features)}",
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "drift_detected": overall_drift,
                "feature_drift_results": drift_results,
                "reference_period": f"{reference_start.date()} to {reference_end.date()}",
                "current_period": f"{current_start.date()} to {datetime.now().date()}",
                "reference_samples": len(ref_result.data),
                "current_samples": len(curr_result.data),
                "drift_threshold": self.drift_threshold,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {"error": str(e)}
    
    async def get_prediction_stats(self, days_back: int = 7) -> Dict[str, Any]:
        """Get prediction statistics for the specified period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            result = supabase.table("model_predictions").select("*").gte(
                "timestamp", cutoff_date.isoformat()
            ).execute()
            
            if not result.data:
                return {
                    "period_days": days_back,
                    "total_predictions": 0,
                    "predictions_per_day": 0
                }
            
            predictions = [record["prediction"] for record in result.data]
            probabilities = [record["probability"] for record in result.data]
            
            # Calculate statistics
            stats_data = {
                "period_days": days_back,
                "total_predictions": len(predictions),
                "predictions_per_day": len(predictions) / days_back,
                "prediction_distribution": {
                    "high_risk_count": sum(1 for p in predictions if p == 1),
                    "low_risk_count": sum(1 for p in predictions if p == 0),
                    "high_risk_percentage": (sum(1 for p in predictions if p == 1) / len(predictions)) * 100
                },
                "probability_stats": {
                    "mean": float(np.mean(probabilities)),
                    "std": float(np.std(probabilities)),
                    "min": float(np.min(probabilities)),
                    "max": float(np.max(probabilities)),
                    "median": float(np.median(probabilities))
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return stats_data
            
        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")
            return {"error": str(e)}

# Global instance
metrics_service = MetricsService()