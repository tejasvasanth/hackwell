#!/usr/bin/env python3
"""
Test script for EvidentlyAI monitoring service
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.monitoring_service import monitoring_service

def test_monitoring_service():
    """Test the monitoring service functionality"""
    print("Testing EvidentlyAI Monitoring Service...")
    
    try:
        # Create sample reference data
        np.random.seed(42)
        n_samples = 1000
        
        reference_data = pd.DataFrame({
            'age': np.random.normal(55, 15, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'blood_pressure': np.random.normal(120, 20, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'exercise_frequency': np.random.poisson(3, n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'gender': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        })
        
        # Calculate risk score and target
        risk_score = (
            reference_data['age'] * 0.02 +
            reference_data['cholesterol'] * 0.01 +
            reference_data['blood_pressure'] * 0.015 +
            reference_data['bmi'] * 0.1 +
            reference_data['diabetes'] * 10 +
            reference_data['smoking'] * 8 +
            reference_data['family_history'] * 5
        )
        
        reference_data['risk_score'] = risk_score
        reference_data['target'] = (risk_score > risk_score.median()).astype(int)
        
        # Add some predictions to reference data (simulated)
        reference_predictions = np.random.uniform(0, 1, n_samples)
        reference_data['prediction'] = reference_predictions
        
        print(f"Created reference data with {len(reference_data)} samples")
        
        # Set reference data
        monitoring_service.set_reference_data(reference_data)
        
        # Create current data with some drift
        current_data = pd.DataFrame({
            'age': np.random.normal(60, 15, 500),  # Age drift
            'cholesterol': np.random.normal(220, 40, 500),  # Cholesterol drift
            'blood_pressure': np.random.normal(125, 20, 500),
            'bmi': np.random.normal(26, 5, 500),
            'exercise_frequency': np.random.poisson(2, 500),  # Exercise drift
            'diabetes': np.random.choice([0, 1], 500, p=[0.75, 0.25]),
            'smoking': np.random.choice([0, 1], 500, p=[0.65, 0.35]),
            'family_history': np.random.choice([0, 1], 500, p=[0.55, 0.45]),
            'gender': np.random.choice([0, 1], 500, p=[0.5, 0.5])
        })
        
        # Calculate risk score and target for current data
        current_risk_score = (
            current_data['age'] * 0.02 +
            current_data['cholesterol'] * 0.01 +
            current_data['blood_pressure'] * 0.015 +
            current_data['bmi'] * 0.1 +
            current_data['diabetes'] * 10 +
            current_data['smoking'] * 8 +
            current_data['family_history'] * 5
        )
        
        current_data['risk_score'] = current_risk_score
        current_data['target'] = (current_risk_score > current_risk_score.median()).astype(int)
        
        # Add some predictions (simulated)
        predictions = np.random.uniform(0, 1, 500)
        current_data['prediction'] = predictions
        
        print(f"Created current data with {len(current_data)} samples")
        
        # Add current data to monitoring service
        monitoring_service.add_current_data(current_data)
        
        # Generate drift report
        print("\nGenerating drift report...")
        report = monitoring_service.generate_drift_report()
        
        # Display results
        print("\n=== DRIFT REPORT ===")
        print(f"Timestamp: {report['timestamp']}")
        print(f"\nData Drift:")
        print(f"  - Dataset drift detected: {report['data_drift']['dataset_drift']}")
        print(f"  - Drift share: {report['data_drift']['drift_share']:.2%}")
        print(f"  - Drifted columns: {report['data_drift']['number_of_drifted_columns']}")
        
        print(f"\nTarget Drift:")
        print(f"  - Drift detected: {report['target_drift']['drift_detected']}")
        print(f"  - Drift score: {report['target_drift']['drift_score']:.3f}")
        
        print(f"\nPrediction Drift:")
        print(f"  - Drift detected: {report['prediction_drift']['drift_detected']}")
        print(f"  - Drift score: {report['prediction_drift']['drift_score']:.3f}")
        
        print(f"\nData Quality:")
        print(f"  - Missing values share: {report['data_quality']['missing_values_share']:.2%}")
        
        print(f"\nOverall Health:")
        print(f"  - Health status: {report['summary']['overall_health']}")
        print(f"  - Risk level: {report['summary']['risk_level']}")
        print(f"  - Risk factors: {report['summary']['risk_factors']}")
        
        # Check alerts
        alerts = monitoring_service.get_alerts()
        print(f"\n=== ALERTS ({len(alerts)}) ===")
        for alert in alerts:
            print(f"  - [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
        
        # Test threshold updates
        print("\n=== TESTING THRESHOLD UPDATES ===")
        current_thresholds = monitoring_service.get_thresholds()
        print(f"Current thresholds: {current_thresholds}")
        
        # Update thresholds
        new_thresholds = {'data_drift': 0.3, 'target_drift': 0.05}
        monitoring_service.update_thresholds(new_thresholds)
        updated_thresholds = monitoring_service.get_thresholds()
        print(f"Updated thresholds: {updated_thresholds}")
        
        print("\n✅ Monitoring service test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing monitoring service: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_monitoring_service()
    sys.exit(0 if success else 1)