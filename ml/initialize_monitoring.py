import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from ml.monitoring_service import monitoring_service
from ml.sample_data_generator import load_monitoring_data

def initialize_monitoring_service():
    """Initialize the monitoring service with sample data"""
    try:
        logger.info("Initializing monitoring service with sample data")
        
        # Load reference and current data
        reference_data, current_data = load_monitoring_data()
        
        # Set reference data
        monitoring_service.set_reference_data(reference_data, target_column='target')
        logger.info(f"Reference data loaded: {len(reference_data)} samples")
        
        # Add current data with some predictions
        # Generate mock predictions for current data
        np.random.seed(42)
        predictions = np.random.random(len(current_data))
        
        monitoring_service.add_current_data(current_data, predictions=predictions)
        logger.info(f"Current data loaded: {len(current_data)} samples with predictions")
        
        # Test drift report generation
        logger.info("Generating initial drift report...")
        report = monitoring_service.generate_drift_report()
        
        logger.info("Drift report generated successfully!")
        logger.info(f"Dataset drift detected: {report['data_drift']['dataset_drift']}")
        logger.info(f"Drift share: {report['data_drift']['drift_share']:.2%}")
        logger.info(f"Number of drifted features: {report['data_drift']['number_of_drifted_columns']}")
        logger.info(f"Overall health: {report['summary']['overall_health']}")
        logger.info(f"Risk level: {report['summary']['risk_level']}")
        
        # Display alerts if any
        alerts = monitoring_service.get_alerts()
        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts:")
            for alert in alerts:
                logger.warning(f"  - {alert['type']}: {alert['message']}")
        else:
            logger.info("No alerts generated")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring service: {str(e)}")
        return False

def test_monitoring_endpoints():
    """Test monitoring functionality"""
    try:
        logger.info("Testing monitoring service functionality...")
        
        # Test getting latest report
        latest_report = monitoring_service.get_latest_report()
        if latest_report:
            logger.info(f"Latest report timestamp: {latest_report['timestamp']}")
        else:
            logger.warning("No latest report found")
        
        # Test getting alerts
        alerts = monitoring_service.get_alerts()
        logger.info(f"Total alerts: {len(alerts)}")
        
        # Test getting thresholds
        thresholds = monitoring_service.get_thresholds()
        logger.info(f"Current thresholds: {thresholds}")
        
        # Test updating thresholds
        new_thresholds = {'data_drift': 0.3}
        monitoring_service.update_thresholds(new_thresholds)
        logger.info(f"Updated thresholds: {new_thresholds}")
        
        # Test regenerating report with new thresholds
        logger.info("Regenerating report with updated thresholds...")
        new_report = monitoring_service.generate_drift_report()
        
        new_alerts = monitoring_service.get_alerts()
        logger.info(f"Alerts after threshold update: {len(new_alerts)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Monitoring test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Initialize monitoring service
    success = initialize_monitoring_service()
    
    if success:
        logger.info("Monitoring service initialized successfully!")
        
        # Run tests
        test_success = test_monitoring_endpoints()
        
        if test_success:
            logger.info("All monitoring tests passed!")
        else:
            logger.error("Some monitoring tests failed")
    else:
        logger.error("Failed to initialize monitoring service")