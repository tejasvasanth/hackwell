#!/usr/bin/env python3
"""
Test script for monitoring API endpoints
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.monitoring_service import monitoring_service

def setup_test_data():
    """Setup test data for monitoring service"""
    print("Setting up test data...")
    
    # Create reference data
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
    reference_data['prediction'] = np.random.uniform(0, 1, n_samples)
    
    # Set reference data
    monitoring_service.set_reference_data(reference_data)
    
    # Create current data with drift
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
    current_data['prediction'] = np.random.uniform(0, 1, 500)
    
    # Add current data
    monitoring_service.add_current_data(current_data)
    
    print(f"✅ Test data setup complete: {len(reference_data)} reference, {len(current_data)} current samples")

def test_monitoring_service_directly():
    """Test monitoring service directly without API"""
    print("\n=== TESTING MONITORING SERVICE DIRECTLY ===")
    
    try:
        # Test drift report generation
        print("\n1. Testing drift report generation...")
        report = monitoring_service.generate_drift_report()
        print(f"✅ Drift report generated successfully")
        print(f"   - Timestamp: {report['timestamp']}")
        print(f"   - Health status: {report['summary']['overall_health']}")
        print(f"   - Data drift: {report['data_drift']['drift_share']:.2%}")
        print(f"   - Alerts: {len(report['alerts'])}")
        
        # Test status
        print("\n2. Testing monitoring status...")
        latest_report = monitoring_service.get_latest_report()
        alerts = monitoring_service.get_alerts()
        thresholds = monitoring_service.get_thresholds()
        
        status = {
            "is_active": True,
            "reference_data_loaded": monitoring_service._reference_data is not None,
            "current_data_loaded": monitoring_service._current_data is not None,
            "last_report_time": latest_report['timestamp'] if latest_report else None,
            "total_alerts": len(alerts),
            "thresholds": thresholds
        }
        
        print(f"✅ Status retrieved successfully")
        print(f"   - Reference data loaded: {status['reference_data_loaded']}")
        print(f"   - Current data loaded: {status['current_data_loaded']}")
        print(f"   - Total alerts: {status['total_alerts']}")
        
        # Test threshold updates
        print("\n3. Testing threshold updates...")
        original_thresholds = monitoring_service.get_thresholds()
        new_thresholds = {'data_drift': 0.3, 'target_drift': 0.05}
        monitoring_service.update_thresholds(new_thresholds)
        updated_thresholds = monitoring_service.get_thresholds()
        
        print(f"✅ Thresholds updated successfully")
        print(f"   - Original: {original_thresholds}")
        print(f"   - Updated: {updated_thresholds}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing monitoring service: {str(e)}")
        return False

def test_api_endpoints():
    """Test monitoring API endpoints"""
    print("\n=== TESTING API ENDPOINTS ===")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test monitoring status endpoint
        print("\n1. Testing /monitoring/status endpoint...")
        response = requests.get(f"{base_url}/monitoring/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Status endpoint working")
            print(f"   - Response: {json.dumps(status_data, indent=2)}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code} - {response.text}")
            return False
        
        # Test drift report endpoint
        print("\n2. Testing /monitoring/report endpoint...")
        response = requests.get(f"{base_url}/monitoring/report")
        if response.status_code == 200:
            report_data = response.json()
            print(f"✅ Report endpoint working")
            print(f"   - Timestamp: {report_data.get('timestamp')}")
            print(f"   - Health: {report_data.get('summary', {}).get('overall_health')}")
        else:
            print(f"❌ Report endpoint failed: {response.status_code} - {response.text}")
            return False
        
        # Test alerts endpoint
        print("\n3. Testing /monitoring/alerts endpoint...")
        response = requests.get(f"{base_url}/monitoring/alerts")
        if response.status_code == 200:
            alerts_data = response.json()
            print(f"✅ Alerts endpoint working")
            print(f"   - Total alerts: {alerts_data.get('total_count', 0)}")
        else:
            print(f"❌ Alerts endpoint failed: {response.status_code} - {response.text}")
            return False
        
        # Test threshold update endpoint
        print("\n4. Testing /monitoring/thresholds endpoint...")
        threshold_data = {"data_drift": 0.4, "target_drift": 0.08}
        response = requests.post(f"{base_url}/monitoring/thresholds", json=threshold_data)
        if response.status_code == 200:
            update_response = response.json()
            print(f"✅ Threshold update endpoint working")
            print(f"   - Response: {update_response.get('message')}")
        else:
            print(f"❌ Threshold update failed: {response.status_code} - {response.text}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure FastAPI server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API endpoints: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Monitoring API Tests")
    print("=" * 50)
    
    # Setup test data
    setup_test_data()
    
    # Test monitoring service directly
    service_success = test_monitoring_service_directly()
    
    # Test API endpoints
    api_success = test_api_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"   - Monitoring Service: {'✅ PASS' if service_success else '❌ FAIL'}")
    print(f"   - API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if service_success and api_success:
        print("\n🎉 All monitoring tests passed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()