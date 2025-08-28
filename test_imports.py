import sys
sys.path.append('.')

try:
    print("Testing imports...")
    
    print("Importing config.settings...")
    from config.settings import settings
    print("✓ config.settings imported successfully")
    
    print("Importing ml.monitoring_service...")
    from ml.monitoring_service import monitoring_service
    print("✓ ml.monitoring_service imported successfully")
    
    print("Importing ml.mlflow_service...")
    from ml.mlflow_service import mlflow_service
    print("✓ ml.mlflow_service imported successfully")
    
    print("Importing api.main...")
    from api.main import app
    print("✓ api.main imported successfully")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()