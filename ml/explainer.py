import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import json
import base64
import io
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelExplainer:
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = None
        self.training_data = None
        self.model = None
        self.feature_descriptions = {
            'age': 'Patient age in years',
            'gender_encoded': 'Gender (0=Female, 1=Male)',
            'bmi': 'Body Mass Index',
            'smoking': 'Smoking status (0=No, 1=Yes)',
            'exercise': 'Exercise frequency (0=Low, 1=Medium, 2=High)',
            'steps': 'Daily steps count',
            'sleep_hours': 'Average sleep hours per night',
            'heart_rate': 'Resting heart rate (bpm)',
            'activity_level': 'Physical activity level (1-5 scale)',
            'stress_level': 'Stress level (1-5 scale)',
            'systolic_bp': 'Systolic blood pressure (mmHg)',
            'diastolic_bp': 'Diastolic blood pressure (mmHg)',
            'total_cholesterol': 'Total cholesterol (mg/dL)',
            'ldl_cholesterol': 'LDL cholesterol (mg/dL)',
            'hdl_cholesterol': 'HDL cholesterol (mg/dL)',
            'glucose': 'Fasting glucose (mg/dL)',
            'hba1c': 'HbA1c percentage'
        }
    
    def initialize_explainers(self, model, training_data: pd.DataFrame, mode: str = 'classification'):
        """Initialize SHAP and LIME explainers with training data"""
        try:
            self.model = model
            self.training_data = training_data
            self.feature_names = training_data.columns.tolist()
            
            # Initialize SHAP explainer - try TreeExplainer first, fallback to KernelExplainer
            try:
                self.shap_explainer = shap.TreeExplainer(model)
                logger.info("SHAP TreeExplainer initialized")
            except Exception as tree_error:
                logger.warning(f"TreeExplainer failed: {tree_error}. Using KernelExplainer instead.")
                # Use a smaller sample for KernelExplainer to improve performance
                background_sample = training_data.sample(min(100, len(training_data)), random_state=42)
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    background_sample.values
                )
                logger.info("SHAP KernelExplainer initialized")
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data.values,
                feature_names=self.feature_names,
                mode=mode,
                class_names=['Low Risk', 'High Risk'] if mode == 'classification' else None,
                discretize_continuous=True,
                random_state=42
            )
            logger.info("LIME explainer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing explainers: {e}")
            raise
    
    async def explain_prediction(self, features: Dict[str, Any], 
                               prediction: Optional[float] = None,
                               patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive explanations using both SHAP and LIME"""
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Get prediction if not provided
            if prediction is None and self.model is not None:
                prediction_proba = self.model.predict_proba(feature_df)[0]
                prediction = float(prediction_proba[1])  # Probability of high risk
                prediction_class = int(self.model.predict(feature_df)[0])
            else:
                prediction_class = 1 if prediction > 0.5 else 0
            
            explanations = {
                "patient_id": patient_id,
                "timestamp": datetime.now().isoformat(),
                "prediction": {
                    "probability": float(prediction) if prediction else None,
                    "class": prediction_class,
                    "risk_level": "High Risk" if prediction_class == 1 else "Low Risk",
                    "confidence": abs(prediction - 0.5) * 2 if prediction else None
                },
                "patient_features": self._format_patient_features(features),
                "shap_explanation": None,
                "lime_explanation": None,
                "summary": None
            }
            
            # SHAP explanation
            if self.shap_explainer is not None:
                shap_values = self.shap_explainer.shap_values(feature_df)
                logger.debug(f"SHAP values type: {type(shap_values)}, shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'no shape'}")
                
                # Handle binary classification (get positive class SHAP values)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_vals = shap_values[1]  # Positive class
                    logger.debug(f"Using positive class SHAP values, shape: {shap_vals.shape}")
                else:
                    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                    logger.debug(f"Using SHAP values, shape: {shap_vals.shape if hasattr(shap_vals, 'shape') else 'no shape'}")
                
                # Handle expected_value properly for different explainer types
                if isinstance(self.shap_explainer.expected_value, np.ndarray):
                    if len(self.shap_explainer.expected_value) > 1:
                        base_value = float(self.shap_explainer.expected_value[1])  # Binary classification positive class
                    else:
                        base_value = float(self.shap_explainer.expected_value[0])
                else:
                    base_value = float(self.shap_explainer.expected_value)
                
                # Ensure shap_vals is properly shaped for calculations
                if shap_vals.ndim == 3:  # (1, n_features, n_classes)
                    shap_vals_calc = shap_vals[0, :, 1]  # First sample, all features, positive class
                elif shap_vals.ndim == 2:  # (n_features, n_classes) or (1, n_features)
                    if shap_vals.shape[1] == 2:  # Binary classification
                        shap_vals_calc = shap_vals[:, 1]  # Positive class
                    else:
                        shap_vals_calc = shap_vals[0]  # Single sample
                else:
                    shap_vals_calc = shap_vals  # Already 1D
                
                logger.debug(f"SHAP values for calculation, shape: {shap_vals_calc.shape}, values: {shap_vals_calc[:5]}")
                
                shap_explanation = {
                    "method": "SHAP (SHapley Additive exPlanations)",
                    "base_value": base_value,
                    "feature_contributions": self._get_shap_contributions(features, shap_vals_calc),
                    "top_positive_features": self._get_top_features(features, shap_vals_calc, positive=True, top_k=5),
                    "top_negative_features": self._get_top_features(features, shap_vals_calc, positive=False, top_k=5),
                    "total_impact": float(np.sum(shap_vals_calc)),
                    "waterfall_data": self._create_waterfall_data(features, shap_vals_calc)
                }
                explanations["shap_explanation"] = shap_explanation
            
            # LIME explanation
            if self.lime_explainer is not None:
                lime_exp = self.lime_explainer.explain_instance(
                    feature_df.values[0],
                    self._predict_fn,
                    num_features=len(self.feature_names),
                    labels=[1]  # Explain positive class
                )
                
                lime_explanation = {
                    "method": "LIME (Local Interpretable Model-agnostic Explanations)",
                    "local_prediction": float(lime_exp.predict_proba[1]),
                    "feature_weights": dict(lime_exp.as_list(label=1)),
                    "model_score": float(lime_exp.score),
                    "intercept": float(lime_exp.intercept[1]),
                    "top_features": self._format_lime_features(lime_exp.as_list(label=1)),
                    "local_model_r2": float(lime_exp.score) if lime_exp.score else None
                }
                explanations["lime_explanation"] = lime_explanation
            
            # Generate summary
            explanations["explanation_summary"] = self._generate_explanation_summary(explanations)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _format_patient_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Format patient features with descriptions"""
        formatted_features = {}
        for feature, value in features.items():
            formatted_features[feature] = {
                "value": value,
                "description": self.feature_descriptions.get(feature, f"Feature: {feature}"),
                "formatted_value": self._format_feature_value(feature, value)
            }
        return formatted_features
    
    def _format_feature_value(self, feature: str, value: Any) -> str:
        """Format feature values for display"""
        if feature == 'gender_encoded':
            return "Male" if value == 1 else "Female"
        elif feature == 'smoking':
            return "Yes" if value == 1 else "No"
        elif feature == 'exercise':
            exercise_levels = {0: "Low", 1: "Medium", 2: "High"}
            return exercise_levels.get(value, str(value))
        elif feature in ['bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol', 'glucose', 'hba1c']:
            return f"{value:.1f}"
        elif feature == 'steps':
            return f"{int(value):,}"
        else:
            return str(value)
    
    def _get_shap_contributions(self, features: Dict[str, Any], shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Get detailed SHAP contributions for each feature"""
        contributions = []
        
        # Ensure shap_values is 1D array for single prediction
        if shap_values.ndim > 1:
            shap_vals_1d = shap_values[0]  # Take first row for single prediction
        else:
            shap_vals_1d = shap_values
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(shap_vals_1d):  # Safety check
                shap_val = float(shap_vals_1d[i])
                contribution = {
                    "feature": feature_name,
                    "feature_value": features.get(feature_name, 0),
                    "formatted_value": self._format_feature_value(feature_name, features.get(feature_name, 0)),
                    "shap_value": shap_val,
                    "abs_shap_value": float(abs(shap_val)),
                    "contribution_type": "increases_risk" if shap_val > 0 else "decreases_risk",
                    "description": self.feature_descriptions.get(feature_name, f"Feature: {feature_name}"),
                    "impact_strength": self._get_impact_strength(abs(shap_val), shap_vals_1d)
                }
                contributions.append(contribution)
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)
        
        return contributions
    
    def _get_impact_strength(self, abs_shap_value: float, all_shap_values: np.ndarray) -> str:
        """Categorize the strength of feature impact"""
        max_impact = np.max(np.abs(all_shap_values))
        relative_impact = abs_shap_value / max_impact if max_impact > 0 else 0
        
        if relative_impact >= 0.7:
            return "Very High"
        elif relative_impact >= 0.4:
            return "High"
        elif relative_impact >= 0.2:
            return "Medium"
        elif relative_impact >= 0.1:
            return "Low"
        else:
            return "Very Low"
    
    def _get_top_features(self, features: Dict[str, Any], shap_values: np.ndarray, positive: bool = True, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top features that increase or decrease risk"""
        feature_impacts = []
        
        # Ensure shap_values is 1D array for single prediction
        if shap_values.ndim > 1:
            shap_vals_1d = shap_values[0]
        else:
            shap_vals_1d = shap_values
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(shap_vals_1d):
                shap_val = float(shap_vals_1d[i])
                if (positive and shap_val > 0) or (not positive and shap_val < 0):
                    feature_impacts.append({
                        "feature": feature_name,
                        "shap_value": shap_val,
                        "feature_value": features.get(feature_name, 0),
                        "formatted_value": self._format_feature_value(feature_name, features.get(feature_name, 0)),
                        "description": self.feature_descriptions.get(feature_name, f"Feature: {feature_name}")
                    })
        
        # Sort by absolute SHAP value and return top k
        feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return feature_impacts[:top_k]
    
    def _create_waterfall_data(self, features: Dict[str, Any], shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Create waterfall chart data for SHAP values"""
        # Handle expected_value properly for different explainer types
        if isinstance(self.shap_explainer.expected_value, np.ndarray):
            if len(self.shap_explainer.expected_value) > 1:
                base_value = float(self.shap_explainer.expected_value[1])  # Binary classification positive class
            else:
                base_value = float(self.shap_explainer.expected_value[0])
        else:
            base_value = float(self.shap_explainer.expected_value)
        
        # Ensure shap_values is 1D array for single prediction
        if shap_values.ndim > 1:
            shap_vals_1d = shap_values[0]
        else:
            shap_vals_1d = shap_values
        
        waterfall_data = [{
            "feature": "Base Value",
            "value": base_value,
            "cumulative": base_value,
            "type": "base"
        }]
        
        cumulative = base_value
        for i, feature_name in enumerate(self.feature_names):
            if i < len(shap_vals_1d):
                shap_val = float(shap_vals_1d[i])
                cumulative += shap_val
                
                waterfall_data.append({
                    "feature": feature_name,
                    "value": shap_val,
                    "cumulative": cumulative,
                    "type": "positive" if shap_val > 0 else "negative",
                    "feature_value": self._format_feature_value(feature_name, features.get(feature_name, 0))
                })
        
        return waterfall_data
    
    def _format_lime_features(self, lime_features: List[tuple]) -> List[Dict[str, Any]]:
        """Format LIME features for better display"""
        formatted_features = []
        
        for feature_name, weight in lime_features:
            formatted_features.append({
                "feature": feature_name,
                "weight": float(weight),
                "abs_weight": float(abs(weight)),
                "impact_type": "increases_risk" if weight > 0 else "decreases_risk",
                "description": self.feature_descriptions.get(feature_name, f"Feature: {feature_name}")
            })
        
        # Sort by absolute weight
        formatted_features.sort(key=lambda x: x["abs_weight"], reverse=True)
        return formatted_features
    
    def _generate_explanation_summary(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a human-readable summary of the explanations"""
        summary = {
            "risk_assessment": explanations["prediction"]["risk_level"],
            "confidence": explanations["prediction"]["confidence"],
            "key_insights": [],
            "recommendations": []
        }
        
        # Add SHAP insights
        if explanations["shap_explanation"]:
            shap_data = explanations["shap_explanation"]
            top_positive = shap_data["top_positive_features"][:3]
            top_negative = shap_data["top_negative_features"][:3]
            
            if top_positive:
                summary["key_insights"].append(f"Top risk factors: {', '.join([f['feature'] for f in top_positive])}")
            
            if top_negative:
                summary["key_insights"].append(f"Top protective factors: {', '.join([f['feature'] for f in top_negative])}")
        
        # Add recommendations based on risk factors
        if explanations["prediction"]["risk_level"] == "High Risk":
            summary["recommendations"].extend([
                "Consult with healthcare provider for comprehensive risk assessment",
                "Consider lifestyle modifications based on identified risk factors",
                "Regular monitoring of key health metrics recommended"
            ])
        else:
            summary["recommendations"].extend([
                "Maintain current healthy lifestyle practices",
                "Continue regular health check-ups",
                "Monitor any risk factors that were identified"
            ])
        
        return summary
    
    def _predict_fn(self, X):
        """Prediction function for LIME"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
            
        # Get predictions
        predictions = self.model.predict_proba(X_df)
        return predictions
    
    async def get_global_feature_importance(self, model, 
                                          sample_data: pd.DataFrame,
                                          max_samples: int = 1000) -> Dict[str, Any]:
        """Get comprehensive global feature importance using SHAP"""
        try:
            if self.shap_explainer is None:
                self.initialize_explainers(model, sample_data)
            
            # Limit sample size for performance
            if len(sample_data) > max_samples:
                sample_data = sample_data.sample(n=max_samples, random_state=42)
                logger.info(f"Sampled {max_samples} rows for global analysis")
            
            # Calculate SHAP values for sample data
            shap_values = self.shap_explainer.shap_values(sample_data)
            
            # Handle binary classification
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_vals = shap_values[1]  # Positive class
            else:
                shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            # Calculate various importance metrics
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            mean_shap = shap_vals.mean(axis=0)
            std_shap = shap_vals.std(axis=0)
            
            # Create comprehensive feature importance analysis
            feature_analysis = []
            for i, feature_name in enumerate(self.feature_names):
                analysis = {
                    "feature": feature_name,
                    "description": self.feature_descriptions.get(feature_name, f"Feature: {feature_name}"),
                    "mean_abs_shap": float(mean_abs_shap[i]),
                    "mean_shap": float(mean_shap[i]),
                    "std_shap": float(std_shap[i]),
                    "importance_rank": 0,  # Will be set after sorting
                    "impact_direction": "increases_risk" if mean_shap[i] > 0 else "decreases_risk",
                    "consistency": "high" if std_shap[i] < mean_abs_shap[i] * 0.5 else "medium" if std_shap[i] < mean_abs_shap[i] else "low",
                    "feature_stats": {
                        "min_value": float(sample_data[feature_name].min()),
                        "max_value": float(sample_data[feature_name].max()),
                        "mean_value": float(sample_data[feature_name].mean()),
                        "std_value": float(sample_data[feature_name].std())
                    }
                }
                feature_analysis.append(analysis)
            
            # Sort by mean absolute SHAP value and assign ranks
            feature_analysis.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            for i, feature in enumerate(feature_analysis):
                feature["importance_rank"] = i + 1
            
            # Create summary statistics
            summary_stats = {
                "total_features": len(self.feature_names),
                "total_samples": len(sample_data),
                "base_value": float(self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, np.ndarray) else self.shap_explainer.expected_value),
                "top_5_features": [f["feature"] for f in feature_analysis[:5]],
                "most_consistent_features": [f["feature"] for f in sorted(feature_analysis, key=lambda x: x["std_shap"])[:5]],
                "highest_risk_factors": [f["feature"] for f in sorted([f for f in feature_analysis if f["impact_direction"] == "increases_risk"], key=lambda x: x["mean_shap"], reverse=True)[:5]],
                "strongest_protective_factors": [f["feature"] for f in sorted([f for f in feature_analysis if f["impact_direction"] == "decreases_risk"], key=lambda x: abs(x["mean_shap"]), reverse=True)[:5]]
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "method": "SHAP Global Feature Importance Analysis",
                "feature_analysis": feature_analysis,
                "summary_stats": summary_stats,
                "visualization_data": self._create_global_viz_data(feature_analysis, shap_vals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating global feature importance: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _create_global_viz_data(self, feature_analysis: List[Dict], shap_values: np.ndarray) -> Dict[str, Any]:
        """Create visualization data for global feature importance"""
        try:
            # Bar chart data for feature importance
            bar_chart_data = {
                "features": [f["feature"] for f in feature_analysis[:10]],  # Top 10
                "importance_values": [f["mean_abs_shap"] for f in feature_analysis[:10]],
                "colors": ["#ff6b6b" if f["impact_direction"] == "increases_risk" else "#4ecdc4" for f in feature_analysis[:10]]
            }
            
            # Box plot data for SHAP value distributions
            box_plot_data = []
            for i, feature_name in enumerate(self.feature_names[:10]):  # Top 10 features
                box_plot_data.append({
                    "feature": feature_name,
                    "shap_values": shap_values[:, i].tolist(),
                    "quartiles": {
                        "q1": float(np.percentile(shap_values[:, i], 25)),
                        "median": float(np.percentile(shap_values[:, i], 50)),
                        "q3": float(np.percentile(shap_values[:, i], 75)),
                        "min": float(np.min(shap_values[:, i])),
                        "max": float(np.max(shap_values[:, i]))
                    }
                })
            
            return {
                "bar_chart": bar_chart_data,
                "box_plot": box_plot_data,
                "heatmap_data": self._create_correlation_heatmap_data(shap_values)
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization data: {e}")
            return {"error": str(e)}
    
    def _create_correlation_heatmap_data(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Create correlation heatmap data for SHAP values"""
        try:
            # Calculate correlation matrix of SHAP values
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
            correlation_matrix = shap_df.corr()
            
            return {
                "features": self.feature_names,
                "correlation_matrix": correlation_matrix.values.tolist(),
                "feature_pairs": [
                    {
                        "feature1": self.feature_names[i],
                        "feature2": self.feature_names[j],
                        "correlation": float(correlation_matrix.iloc[i, j])
                    }
                    for i in range(len(self.feature_names))
                    for j in range(i+1, len(self.feature_names))
                    if abs(correlation_matrix.iloc[i, j]) > 0.3  # Only significant correlations
                ]
            }
        except Exception as e:
            logger.error(f"Error creating correlation data: {e}")
            return {"error": str(e)}
    
    async def generate_explanation_plots(self, explanations: Dict[str, Any], 
                                       output_format: str = "base64") -> Dict[str, Any]:
        """Generate visualization plots for explanations"""
        try:
            plots = {}
            
            # SHAP waterfall plot
            if explanations.get("shap_explanation"):
                waterfall_plot = self._create_waterfall_plot(explanations["shap_explanation"]["waterfall_data"])
                plots["shap_waterfall"] = waterfall_plot
                
                # SHAP feature importance plot
                importance_plot = self._create_feature_importance_plot(explanations["shap_explanation"]["feature_contributions"])
                plots["shap_importance"] = importance_plot
            
            # LIME feature weights plot
            if explanations.get("lime_explanation"):
                lime_plot = self._create_lime_plot(explanations["lime_explanation"]["top_features"])
                plots["lime_weights"] = lime_plot
            
            return {
                "plots": plots,
                "format": output_format,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation plots: {e}")
            return {"error": str(e)}
    
    def _create_waterfall_plot(self, waterfall_data: List[Dict]) -> str:
        """Create waterfall plot as base64 encoded image"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            features = [item["feature"] for item in waterfall_data[1:]]  # Skip base value
            values = [item["value"] for item in waterfall_data[1:]]
            cumulative = [item["cumulative"] for item in waterfall_data]
            
            # Create waterfall chart
            colors = ['green' if v < 0 else 'red' for v in values]
            bars = ax.bar(range(len(features)), values, color=colors, alpha=0.7)
            
            # Add base value line
            ax.axhline(y=waterfall_data[0]["value"], color='blue', linestyle='--', alpha=0.7, label='Base Value')
            
            # Customize plot
            ax.set_xlabel('Features')
            ax.set_ylabel('SHAP Value Contribution')
            ax.set_title('SHAP Waterfall Plot - Feature Contributions')
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_base64
            
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            return ""
    
    def _create_feature_importance_plot(self, contributions: List[Dict]) -> str:
        """Create feature importance plot as base64 encoded image"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get top 10 features
            top_features = contributions[:10]
            features = [item["feature"] for item in top_features]
            abs_values = [item["abs_shap_value"] for item in top_features]
            colors = ['red' if item["contribution_type"] == "increases_risk" else 'green' for item in top_features]
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(features)), abs_values, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_xlabel('Absolute SHAP Value')
            ax.set_ylabel('Features')
            ax.set_title('Top 10 Feature Importance (SHAP)')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7, label="Increases Risk")
            green_patch = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.7, label="Decreases Risk")
            ax.legend(handles=[red_patch, green_patch])
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_base64
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return ""
    
    def _create_lime_plot(self, lime_features: List[Dict]) -> str:
        """Create LIME feature weights plot as base64 encoded image"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            features = [item["feature"] for item in lime_features[:10]]
            weights = [item["weight"] for item in lime_features[:10]]
            colors = ['red' if w > 0 else 'green' for w in weights]
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(features)), weights, color=colors, alpha=0.7)
            
            # Customize plot
            ax.set_xlabel('LIME Weight')
            ax.set_ylabel('Features')
            ax.set_title('LIME Feature Weights')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7, label="Increases Risk")
            green_patch = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.7, label="Decreases Risk")
            ax.legend(handles=[red_patch, green_patch])
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_base64
            
        except Exception as e:
            logger.error(f"Error creating LIME plot: {e}")
            return ""
    
    def export_explanations_json(self, explanations: Dict[str, Any], 
                               file_path: Optional[str] = None) -> str:
        """Export explanations to JSON format"""
        try:
            json_data = json.dumps(explanations, indent=2, ensure_ascii=False)
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                logger.info(f"Explanations exported to {file_path}")
            
            return json_data
            
        except Exception as e:
            logger.error(f"Error exporting explanations: {e}")
            return json.dumps({"error": str(e)}, indent=2)