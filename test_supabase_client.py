#!/usr/bin/env python3
"""
Test script for the enhanced Supabase client with healthcare-specific functions.
This script demonstrates the usage of patient data retrieval, ECG file upload, and prediction logging.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from config.supabase_client import SupabaseClient

def test_supabase_client():
    """Test the enhanced Supabase client functions."""
    
    # Initialize client
    try:
        client = SupabaseClient()
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        return
    
    # Test 1: Get patient data
    print("\n🔍 Testing patient data retrieval...")
    try:
        patient_id = 1  # Test with patient ID 1
        patient_data = client.get_patient_data(patient_id)
        print(f"✅ Patient data retrieved: {patient_data}")
    except Exception as e:
        print(f"❌ Failed to retrieve patient data: {e}")
    
    # Test 2: Upload ECG file (create a dummy file for testing)
    print("\n📁 Testing ECG file upload...")
    try:
        # Create a dummy ECG file for testing
        test_ecg_path = "test_ecg.txt"
        with open(test_ecg_path, 'w') as f:
            f.write("Dummy ECG data for testing purposes\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write("Lead I: 0.5, 0.3, 0.8, 0.2, -0.1\n")
            f.write("Lead II: 0.7, 0.4, 0.9, 0.3, 0.0\n")
        
        patient_id = 1
        public_url = client.upload_ecg(patient_id, test_ecg_path)
        print(f"✅ ECG file uploaded successfully: {public_url}")
        
        # Clean up test file
        os.remove(test_ecg_path)
        
    except Exception as e:
        print(f"❌ Failed to upload ECG file: {e}")
        # Clean up test file if it exists
        if os.path.exists(test_ecg_path):
            os.remove(test_ecg_path)
    
    # Test 3: Log prediction
    print("\n📊 Testing prediction logging...")
    try:
        patient_id = 1
        prediction = "high_risk"
        probability = 0.85
        timestamp = datetime.now()
        
        result = client.log_prediction(patient_id, prediction, probability, timestamp)
        print(f"✅ Prediction logged successfully: {result}")
        
    except Exception as e:
        print(f"❌ Failed to log prediction: {e}")
    
    print("\n🎉 Supabase client testing completed!")
    print("\n📝 Note: Some tests may fail if the database tables don't exist yet.")
    print("   Please ensure the following tables are created in your Supabase database:")
    print("   - patients (id, name, age, gender, smoking, exercise, bp, cholesterol, diabetes_status)")
    print("   - predictions (id, patient_id, prediction, probability, timestamp)")
    print("   - Storage bucket 'ecg_files' should be created and configured")

def main():
    """Main function to run the tests."""
    print("🚀 Starting Supabase Client Tests")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("❌ Missing environment variables: SUPABASE_URL and SUPABASE_KEY must be set")
        print("   Please create a .env file with your Supabase credentials")
        return
    
    test_supabase_client()

if __name__ == "__main__":
    main()