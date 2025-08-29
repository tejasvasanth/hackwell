#!/usr/bin/env python3
"""
Test script for Supabase-integrated monitoring and sample data generation
"""

import sys
import os
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.sample_data_generator import store_sample_data_in_supabase
from ml.monitoring_service import monitoring_service
from supabase_client import get_supabase_client

def test_sample_data_generation():
    """Test generating and storing sample data in Supabase"""
    logger.info("=== Testing Sample Data Generation ===")
    
    try:
        # Generate sample data
        success = store_sample_data_in_supabase(
            n_patients=20,
            n_lab_results_per_patient=5,
            n_lifestyle_records_per_patient=3
        )
        
        if success:
            logger.info("✅ Sample data generation successful")
            return True
        else:
            logger.error("❌ Sample data generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in sample data generation: {e}")
        return False

def test_monitoring_with_supabase():
    """Test monitoring service with Supabase data"""
    logger.info("=== Testing Monitoring with Supabase ===")
    
    try:
        # Load data from Supabase
        reference_data, current_data = monitoring_service.load_data_from_supabase(limit=100)
        
        if reference_data is None or current_data is None:
            logger.warning("No data available in Supabase for monitoring")
            return False
        
        logger.info(f"Loaded reference data: {len(reference_data)} samples")
        logger.info(f"Loaded current data: {len(current_data)} samples")
        
        # Set reference data
        monitoring_service.set_reference_data(reference_data)
        
        # Generate monitoring report
        report = monitoring_service.generate_drift_report(current_data)
        
        if report:
            logger.info("✅ Monitoring report generated successfully")
            logger.info(f"Data drift detected: {report.get('data_drift', {}).get('dataset_drift', False)}")
            logger.info(f"Overall health: {report.get('summary', {}).get('overall_health', 'unknown')}")
            
            # Check alerts
            alerts = monitoring_service.get_alerts()
            logger.info(f"Active alerts: {len(alerts)}")
            
            for alert in alerts[-3:]:  # Show last 3 alerts
                logger.info(f"  - [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
            
            return True
        else:
            logger.error("❌ Failed to generate monitoring report")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in monitoring test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_supabase_connection():
    """Test basic Supabase connection"""
    logger.info("=== Testing Supabase Connection ===")
    
    try:
        supabase = get_supabase_client()
        
        # Test basic query
        response = supabase.table('patients').select('id').limit(1).execute()
        
        if response.data:
            logger.info("✅ Supabase connection successful")
            return True
        else:
            logger.warning("⚠️ Supabase connected but no patient data found")
            return True
            
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting Supabase monitoring integration tests...")
    
    results = {
        'connection': test_supabase_connection(),
        'sample_data': test_sample_data_generation(),
        'monitoring': test_monitoring_with_supabase()
    }
    
    logger.info("\n=== Test Results ===")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Supabase monitoring integration is working.")
    else:
        logger.error("\n💥 Some tests failed. Check the logs above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)