#!/usr/bin/env python3
"""
Simplified test script for the ModelExplainer class to verify SHAP and LIME implementations
This version doesn't require scikit-learn and focuses on testing the core functionality
"""

import asyncio
import pandas as pd
import numpy as np
import json
from explainer import ModelExplainer
from loguru import logger

class MockModel:
    """
    Mock model for testing that mimics XGBoost interface
    """
    def __init__(self):
        self.feature_names = [
            'age', 'gender', 'height', 'weight', 'systolic_bp', 'diastolic_bp',
            'cholesterol', 'glucose', 'smoking', 'alcohol', 'physical_activity',
            'bmi', 'pulse_pressure', 'map', 'cholesterol_ratio', 'glucose_category',
            'bp_category'
        ]
        # Mock weights for prediction
        self.weights = np.random.randn(len(self.feature_names))
    
    def predict_proba(self, X):
        """
        Mock prediction that returns probabilities for both classes
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Simple linear combination for mock prediction
        scores = np.dot(X_array, self.weights)
        # Convert to probabilities using sigmoid
        probs_positive = 1 / (1 + np.exp(-scores))
        probs_negative = 1 - probs_positive
        
        return np.column_stack([probs_negative, probs_positive])
    
    def predict(self, X):
        """
        Mock binary prediction
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

def create_sample_cardiovascular_data(n_samples=100):
    """
    Create sample cardiovascular risk data for testing
    """
    np.random.seed(42)
    
    # Define feature names matching the cardiovascular model
    feature_names = [
        'age', 'gender', 'height', 'weight', 'systolic_bp', 'diastolic_bp',
        'cholesterol', 'glucose', 'smoking', 'alcohol', 'physical_activity',
        'bmi', 'pulse_pressure', 'map', 'cholesterol_ratio', 'glucose_category',
        'bp_category'
    ]
    
    # Generate realistic cardiovascular data
    data = {}
    data['age'] = np.random.normal(55, 15, n_samples).clip(30, 80)
    data['gender'] = np.random.binomial(1, 0.5, n_samples)
    data['height'] = np.random.normal(170, 10, n_samples).clip(150, 200)
    data['weight'] = np.random.normal(75, 15, n_samples).clip(50, 120)
    data['systolic_bp'] = np.random.normal(130, 20, n_samples).clip(90, 200)
    data['diastolic_bp'] = np.random.normal(85, 15, n_samples).clip(60, 120)
    data['cholesterol'] = np.random.normal(220, 50, n_samples).clip(150, 400)
    data['glucose'] = np.random.normal(100, 30, n_samples).clip(70, 200)
    data['smoking'] = np.random.binomial(1, 0.3, n_samples)
    data['alcohol'] = np.random.binomial(1, 0.4, n_samples)
    data['physical_activity'] = np.random.binomial(1, 0.6, n_samples)
    
    # Calculate derived features
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    data['pulse_pressure'] = data['systolic_bp'] - data['diastolic_bp']
    data['map'] = (data['systolic_bp'] + 2 * data['diastolic_bp']) / 3
    data['cholesterol_ratio'] = data['cholesterol'] / 200
    data['glucose_category'] = (data['glucose'] > 126).astype(int)
    data['bp_category'] = ((data['systolic_bp'] > 140) | (data['diastolic_bp'] > 90)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable based on risk factors
    risk_score = (
        (df['age'] - 50) * 0.02 +
        df['smoking'] * 0.3 +
        df['bp_category'] * 0.4 +
        df['glucose_category'] * 0.3 +
        (df['bmi'] - 25) * 0.05
    )
    y = (risk_score + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
    
    return df, y

async def test_explainer_functionality():
    """
    Test the core functionality of ModelExplainer
    """
    logger.info("Testing ModelExplainer functionality...")
    
    try:
        # Create sample data
        logger.info("Creating sample cardiovascular data...")
        X, y = create_sample_cardiovascular_data(100)
        
        # Create mock model
        logger.info("Creating mock model...")
        model = MockModel()
        
        # Test model predictions
        sample_predictions = model.predict_proba(X.head(5))
        logger.info(f"Mock model predictions shape: {sample_predictions.shape}")
        
        # Initialize explainer
        logger.info("Initializing ModelExplainer...")
        explainer = ModelExplainer()
        
        # Test explainer initialization
        try:
            explainer.initialize_explainers(model, X.head(50))  # Use smaller sample
            logger.success("✅ Explainer initialization successful")
        except Exception as e:
            logger.error(f"❌ Explainer initialization failed: {e}")
            return False
        
        # Test single prediction explanation
        logger.info("Testing single prediction explanation...")
        sample_patient = X.iloc[0].to_dict()
        
        try:
            explanation = await explainer.explain_prediction(
                 features=sample_patient,
                 patient_id="test_patient_001"
             )
            logger.success("✅ Single prediction explanation successful")
            
            # Check explanation structure
            required_keys = ['patient_id', 'timestamp', 'prediction', 'shap_explanation', 'lime_explanation']
            missing_keys = [key for key in required_keys if key not in explanation]
            
            if missing_keys:
                logger.warning(f"Missing keys in explanation: {missing_keys}")
            else:
                logger.success("✅ Explanation structure is complete")
                
        except Exception as e:
            logger.error(f"❌ Single prediction explanation failed: {e}")
            return False
        
        # Test global feature importance
        logger.info("Testing global feature importance...")
        try:
            global_importance = await explainer.get_global_feature_importance(
                model=model,
                sample_data=X.head(50)  # Use smaller sample for speed
            )
            logger.success("✅ Global feature importance successful")
            
            # Check global importance structure
            if 'feature_analysis' in global_importance and 'summary_stats' in global_importance:
                logger.success("✅ Global importance structure is complete")
            else:
                logger.warning("⚠️ Global importance structure incomplete")
                
        except Exception as e:
            logger.error(f"❌ Global feature importance failed: {e}")
            return False
        
        # Test JSON export
        logger.info("Testing JSON export...")
        try:
            json_output = explainer.export_explanations_json(explanation)
            json.loads(json_output)  # Validate JSON
            logger.success("✅ JSON export successful")
        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
            return False
        
        # Test explanation plots (without actually generating images)
        logger.info("Testing explanation plots structure...")
        try:
            plots = await explainer.generate_explanation_plots(explanation)
            if 'plots' in plots:
                logger.success("✅ Explanation plots structure successful")
            else:
                logger.warning("⚠️ Explanation plots structure incomplete")
        except Exception as e:
            logger.error(f"❌ Explanation plots failed: {e}")
            return False
        
        logger.success("🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

async def test_json_structure():
    """
    Test the JSON structure for Streamlit integration
    """
    logger.info("Testing JSON structure for Streamlit integration...")
    
    # Create sample data
    X, y = create_sample_cardiovascular_data(50)
    model = MockModel()
    
    # Initialize explainer
    explainer = ModelExplainer()
    explainer.initialize_explainers(model, X.head(30))
    
    # Get explanation
    sample_patient = X.iloc[0].to_dict()
    explanation = await explainer.explain_prediction(
         features=sample_patient,
         patient_id="streamlit_test_001"
     )
    
    # Check JSON structure for Streamlit compatibility
    logger.info("Checking JSON structure...")
    
    # Test serialization
    json_str = json.dumps(explanation, indent=2)
    parsed_back = json.loads(json_str)
    
    # Check key components
    checks = {
        "Has patient_id": "patient_id" in parsed_back,
        "Has timestamp": "timestamp" in parsed_back,
        "Has prediction": "prediction" in parsed_back,
        "Has SHAP explanation": "shap_explanation" in parsed_back,
        "Has LIME explanation": "lime_explanation" in parsed_back,
        "Has explanation summary": "explanation_summary" in parsed_back,
        "Prediction has probability": "probability" in parsed_back.get("prediction", {}),
        "Prediction has risk_level": "risk_level" in parsed_back.get("prediction", {}),
        "SHAP has feature_contributions": "feature_contributions" in parsed_back.get("shap_explanation", {}),
        "LIME has top_features": "top_features" in parsed_back.get("lime_explanation", {})
    }
    
    print("\n=== JSON STRUCTURE VALIDATION ===")
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}: {result}")
    
    all_passed = all(checks.values())
    if all_passed:
        logger.success("🎉 All JSON structure checks passed!")
    else:
        logger.warning("⚠️ Some JSON structure checks failed")
    
    return all_passed

if __name__ == "__main__":
    logger.info("Starting ModelExplainer tests...")
    
    async def run_all_tests():
        # Test core functionality
        functionality_passed = await test_explainer_functionality()
        
        # Test JSON structure
        json_passed = await test_json_structure()
        
        print("\n=== FINAL TEST RESULTS ===")
        print(f"✅ Core Functionality: {'PASSED' if functionality_passed else 'FAILED'}")
        print(f"✅ JSON Structure: {'PASSED' if json_passed else 'FAILED'}")
        
        if functionality_passed and json_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ SHAP global/local feature importance: IMPLEMENTED")
            print("✅ LIME single patient explanations: IMPLEMENTED")
            print("✅ JSON formatting for Streamlit: IMPLEMENTED")
            print("\n🚀 The explainer is ready for Streamlit dashboard integration!")
            return True
        else:
            print("\n❌ SOME TESTS FAILED")
            return False
    
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")
        exit(1)