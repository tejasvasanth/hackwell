#!/usr/bin/env python3
"""
Setup Supabase database schema for Healthcare AI System
"""

import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file")

client = create_client(supabase_url, supabase_key)

def create_sample_data():
    """Create sample data for testing"""
    
    print("Creating sample patients...")
    
    # Sample patients data
    patients = [
        {
            "id": "P001",
            "name": "John Smith",
            "age": 65,
            "gender": "Male",
            "admission_date": "2024-01-15",
            "height": 175,
            "weight": 80,
            "bmi": 26.1,
            "ethnicity": "Caucasian",
            "insurance_type": "Medicare",
            "blood_type": "O+",
            "emergency_contact": "Jane Smith (Wife)",
            "phone": "+1-555-0123",
            "medical_history": ["Hypertension", "Type 2 Diabetes", "Hyperlipidemia"],
            "current_medications": ["Lisinopril 10mg", "Metformin 500mg", "Atorvastatin 20mg"],
            "risk_factors": ["Age > 65", "Diabetes", "Hypertension"]
        },
        {
            "id": "P002",
            "name": "Maria Garcia",
            "age": 58,
            "gender": "Female",
            "admission_date": "2024-01-16",
            "height": 162,
            "weight": 70,
            "bmi": 26.7,
            "ethnicity": "Hispanic",
            "insurance_type": "Private",
            "blood_type": "A+",
            "emergency_contact": "Tom Garcia (Husband)",
            "phone": "+1-555-0456",
            "medical_history": ["Osteoporosis", "Anxiety"],
            "current_medications": ["Alendronate 70mg", "Sertraline 50mg"],
            "risk_factors": ["Postmenopausal", "Family history of heart disease"]
        },
        {
            "id": "P003",
            "name": "Robert Johnson",
            "age": 72,
            "gender": "Male",
            "admission_date": "2024-01-17",
            "height": 180,
            "weight": 95,
            "bmi": 29.3,
            "ethnicity": "African American",
            "insurance_type": "Medicare",
            "blood_type": "B+",
            "emergency_contact": "Lisa Johnson (Daughter)",
            "phone": "+1-555-0789",
            "medical_history": ["Hypertension", "Type 2 Diabetes", "COPD", "Coronary Artery Disease"],
            "current_medications": ["Amlodipine 10mg", "Insulin Glargine", "Albuterol inhaler", "Clopidogrel 75mg"],
            "risk_factors": ["Age > 70", "Diabetes", "Hypertension", "Smoking", "Obesity", "CAD"]
        }
    ]
    
    # Insert patients
    try:
        result = client.table('patients').insert(patients).execute()
        print(f"✓ Inserted {len(patients)} patients")
    except Exception as e:
        print(f"✗ Error inserting patients: {e}")
    
    # Sample lab results
    lab_results = [
        {
            "patient_id": "P001",
            "test_date": "2024-01-20",
            "systolic_bp": 145,
            "diastolic_bp": 90,
            "glucose": 120,
            "cholesterol": 200,
            "hdl": 45,
            "ldl": 160,
            "triglycerides": 180,
            "hemoglobin": 13.5,
            "creatinine": 1.1,
            "hba1c": 6.2
        },
        {
            "patient_id": "P002",
            "test_date": "2024-01-20",
            "systolic_bp": 130,
            "diastolic_bp": 85,
            "glucose": 95,
            "cholesterol": 180,
            "hdl": 55,
            "ldl": 120,
            "triglycerides": 125,
            "hemoglobin": 12.8,
            "creatinine": 0.9,
            "hba1c": 5.8
        },
        {
            "patient_id": "P003",
            "test_date": "2024-01-20",
            "systolic_bp": 160,
            "diastolic_bp": 95,
            "glucose": 140,
            "cholesterol": 240,
            "hdl": 40,
            "ldl": 190,
            "triglycerides": 250,
            "hemoglobin": 11.5,
            "creatinine": 1.3,
            "hba1c": 6.8
        }
    ]
    
    # Insert lab results
    try:
        result = client.table('lab_results').insert(lab_results).execute()
        print(f"✓ Inserted {len(lab_results)} lab results")
    except Exception as e:
        print(f"✗ Error inserting lab results: {e}")
    
    # Sample lifestyle data
    lifestyle_data = [
        {
            "patient_id": "P001",
            "record_date": "2024-01-20",
            "exercise_frequency": 3,
            "smoking_status": "Former",
            "alcohol_consumption": "Moderate",
            "diet_quality": "Good",
            "diet_type": "Mediterranean",
            "sleep_hours": 7,
            "stress_level": "Medium",
            "family_history": True
        },
        {
            "patient_id": "P002",
            "record_date": "2024-01-20",
            "exercise_frequency": 4,
            "smoking_status": "Never",
            "alcohol_consumption": "Light",
            "diet_quality": "Excellent",
            "diet_type": "Vegetarian",
            "sleep_hours": 8,
            "stress_level": "Low",
            "family_history": False
        },
        {
            "patient_id": "P003",
            "record_date": "2024-01-20",
            "exercise_frequency": 1,
            "smoking_status": "Current",
            "alcohol_consumption": "Heavy",
            "diet_quality": "Poor",
            "diet_type": "Standard",
            "sleep_hours": 5,
            "stress_level": "High",
            "family_history": True
        }
    ]
    
    # Insert lifestyle data
    try:
        result = client.table('lifestyle_data').insert(lifestyle_data).execute()
        print(f"✓ Inserted {len(lifestyle_data)} lifestyle records")
    except Exception as e:
        print(f"✗ Error inserting lifestyle data: {e}")
    
    print("\n🎉 Sample data creation completed!")

def test_connection():
    """Test the Supabase connection"""
    try:
        # Try to query patients table
        result = client.table('patients').select('*').limit(1).execute()
        print(f"✓ Connection successful! Found {len(result.data)} records in patients table")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    try:
        print("Testing Supabase connection...")
        if test_connection():
            print("\nCreating sample data...")
            create_sample_data()
        else:
            print("\n❌ Please ensure your Supabase tables are created first.")
            print("You may need to run the SQL schema in the Supabase dashboard.")
        
    except Exception as e:
        print(f"❌ Error: {e}")