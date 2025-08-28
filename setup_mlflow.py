#!/usr/bin/env python3
"""
MLflow Setup Script

Sets up MLflow tracking server and creates necessary experiments
for the cardiovascular risk prediction training pipeline.
"""

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """
    Set up MLflow tracking server and experiments.
    """
    print("🔧 Setting up MLflow for Cardiovascular Risk Prediction")
    print("=" * 60)
    
    try:
        # Create MLflow directory if it doesn't exist
        mlflow_dir = Path("mlruns")
        mlflow_dir.mkdir(exist_ok=True)
        
        # Set tracking URI to local directory
        tracking_uri = f"file://{mlflow_dir.absolute()}"
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        print(f"📍 MLflow tracking URI: {tracking_uri}")
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Create experiments
        experiments = [
            {
                "name": "cardiovascular_risk_prediction",
                "description": "Main experiment for cardiovascular risk prediction model training"
            },
            {
                "name": "model_comparison",
                "description": "Experiment for comparing different model architectures and hyperparameters"
            },
            {
                "name": "feature_engineering",
                "description": "Experiment for testing different feature engineering approaches"
            }
        ]
        
        created_experiments = []
        
        for exp_config in experiments:
            try:
                # Check if experiment already exists
                existing_exp = mlflow.get_experiment_by_name(exp_config["name"])
                
                if existing_exp is None:
                    # Create new experiment
                    exp_id = mlflow.create_experiment(
                        name=exp_config["name"],
                        tags={"description": exp_config["description"]}
                    )
                    created_experiments.append((exp_config["name"], exp_id, "created"))
                    logger.info(f"Created experiment: {exp_config['name']} (ID: {exp_id})")
                else:
                    created_experiments.append((exp_config["name"], existing_exp.experiment_id, "exists"))
                    logger.info(f"Experiment already exists: {exp_config['name']} (ID: {existing_exp.experiment_id})")
                    
            except Exception as e:
                logger.error(f"Error creating experiment {exp_config['name']}: {e}")
        
        # Print summary
        print("\n✅ MLflow Setup Complete!")
        print("\n📊 Experiments:")
        for name, exp_id, status in created_experiments:
            status_icon = "🆕" if status == "created" else "📁"
            print(f"   {status_icon} {name} (ID: {exp_id})")
        
        print("\n🚀 Ready to run training pipeline!")
        print("\n💡 To start MLflow UI, run:")
        print(f"   mlflow ui --backend-store-uri {tracking_uri}")
        print("   Then open: http://localhost:5000")
        
        return True
        
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        return False

def check_mlflow_status():
    """
    Check current MLflow configuration and status.
    """
    print("\n🔍 MLflow Status Check")
    print("-" * 30)
    
    try:
        # Check tracking URI
        current_uri = mlflow.get_tracking_uri()
        print(f"📍 Tracking URI: {current_uri}")
        
        # List experiments
        client = MlflowClient()
        experiments = client.search_experiments()
        
        print(f"\n📊 Available Experiments ({len(experiments)}):")
        for exp in experiments:
            print(f"   • {exp.name} (ID: {exp.experiment_id})")
            if exp.tags:
                desc = exp.tags.get('description', 'No description')
                print(f"     Description: {desc}")
        
        return True
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        print(f"❌ Status check failed: {e}")
        return False

def main():
    """
    Main setup function.
    """
    print("🏥 MLflow Setup for Healthcare ML Pipeline")
    print("=" * 50)
    
    # Setup MLflow
    if setup_mlflow():
        # Check status
        check_mlflow_status()
        
        print("\n" + "=" * 50)
        print("🎯 Next Steps:")
        print("1. Run: python run_training.py")
        print("2. Start MLflow UI: mlflow ui")
        print("3. Open browser: http://localhost:5000")
        print("=" * 50)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()