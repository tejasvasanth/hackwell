#!/usr/bin/env python3
"""
Training Pipeline Runner

Simple script to execute the cardiovascular risk prediction training pipeline.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml.train_cardiovascular_model import main as train_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Execute the training pipeline.
    """
    print("🏥 Cardiovascular Risk Prediction Training Pipeline")
    print("=" * 60)
    print("This pipeline will:")
    print("✓ Pull features from Feast feature store")
    print("✓ Train XGBoost model for cardiovascular risk prediction")
    print("✓ Log comprehensive metrics to MLflow")
    print("✓ Register the best model to MLflow registry")
    print("=" * 60)
    print()
    
    try:
        # Run the training pipeline
        train_main()
        
        print("\n🎉 Training pipeline completed successfully!")
        print("\n📊 Check MLflow UI to view:")
        print("   - Training metrics (accuracy, precision, recall, F1)")
        print("   - Confusion matrix (TP, FP, TN, FN)")
        print("   - Feature importance")
        print("   - Model artifacts")
        print("\n🚀 Model is now registered and ready for deployment!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()