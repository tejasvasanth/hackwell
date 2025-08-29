import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client

def generate_healthcare_data(
    n_samples: int = 1000,
    drift_factor: float = 0.0,
    missing_rate: float = 0.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic healthcare data for monitoring
    
    Args:
        n_samples: Number of samples to generate
        drift_factor: Factor to introduce drift (0.0 = no drift, 1.0 = high drift)
        missing_rate: Rate of missing values to introduce
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with healthcare features and target
    """
    np.random.seed(random_seed)
    
    # Base distributions
    age_mean = 55 + (drift_factor * 10)  # Drift in age distribution
    cholesterol_mean = 200 + (drift_factor * 50)  # Drift in cholesterol
    bp_mean = 120 + (drift_factor * 20)  # Drift in blood pressure
    bmi_mean = 25 + (drift_factor * 5)  # Drift in BMI
    
    # Generate features
    data = {
        'age': np.random.normal(age_mean, 15, n_samples).clip(18, 100),
        'cholesterol': np.random.normal(cholesterol_mean, 40, n_samples).clip(100, 400),
        'blood_pressure': np.random.normal(bp_mean, 20, n_samples).clip(80, 200),
        'bmi': np.random.normal(bmi_mean, 4, n_samples).clip(15, 50),
        'exercise_frequency': np.random.poisson(2 - drift_factor, n_samples).clip(0, 7),
    }
    
    # Categorical features with potential drift
    diabetes_prob = 0.15 + (drift_factor * 0.1)
    smoking_prob = 0.25 - (drift_factor * 0.05)
    family_history_prob = 0.3 + (drift_factor * 0.1)
    
    data.update({
        'diabetes': np.random.binomial(1, diabetes_prob, n_samples),
        'smoking': np.random.binomial(1, smoking_prob, n_samples),
        'family_history': np.random.binomial(1, family_history_prob, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5])
    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable based on features
    risk_score = (
        (df['age'] - 30) * 0.02 +
        (df['cholesterol'] - 150) * 0.003 +
        (df['blood_pressure'] - 100) * 0.01 +
        (df['bmi'] - 20) * 0.05 +
        df['diabetes'] * 0.3 +
        df['smoking'] * 0.2 +
        df['family_history'] * 0.15 +
        (7 - df['exercise_frequency']) * 0.05 +
        np.random.normal(0, 0.1, n_samples)  # Add noise
    )
    
    # Convert to probability and binary target
    risk_probability = 1 / (1 + np.exp(-risk_score))  # Sigmoid
    df['risk_score'] = risk_probability
    df['target'] = (risk_probability > 0.5).astype(int)
    
    # Add missing values if requested
    if missing_rate > 0:
        for col in ['cholesterol', 'blood_pressure', 'bmi', 'exercise_frequency']:
            missing_mask = np.random.random(n_samples) < missing_rate
            df.loc[missing_mask, col] = np.nan
    
    # Add timestamp
    base_time = datetime.now() - timedelta(days=30)
    df['timestamp'] = [
        base_time + timedelta(minutes=i * 30) 
        for i in range(n_samples)
    ]
    
    return df

def generate_reference_and_current_data(
    reference_samples: int = 1000,
    current_samples: int = 500,
    drift_factor: float = 0.3,
    missing_rate_current: float = 0.02
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate reference and current datasets for drift monitoring
    
    Args:
        reference_samples: Number of reference samples
        current_samples: Number of current samples
        drift_factor: Amount of drift to introduce in current data
        missing_rate_current: Missing value rate in current data
    
    Returns:
        Tuple of (reference_data, current_data)
    """
    logger.info(f"Generating reference data with {reference_samples} samples")
    reference_data = generate_healthcare_data(
        n_samples=reference_samples,
        drift_factor=0.0,  # No drift in reference
        missing_rate=0.0,  # No missing values in reference
        random_seed=42
    )
    
    logger.info(f"Generating current data with {current_samples} samples and drift factor {drift_factor}")
    current_data = generate_healthcare_data(
        n_samples=current_samples,
        drift_factor=drift_factor,
        missing_rate=missing_rate_current,
        random_seed=123  # Different seed for current data
    )
    
    return reference_data, current_data

def save_monitoring_datasets(output_dir: str = "data/monitoring"):
    """Generate and save datasets for monitoring testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets with different drift scenarios
    scenarios = [
        {"name": "no_drift", "drift_factor": 0.0, "missing_rate": 0.0},
        {"name": "low_drift", "drift_factor": 0.2, "missing_rate": 0.01},
        {"name": "medium_drift", "drift_factor": 0.5, "missing_rate": 0.03},
        {"name": "high_drift", "drift_factor": 0.8, "missing_rate": 0.05},
    ]
    
    # Generate reference data (same for all scenarios)
    logger.info("Generating reference dataset")
    reference_data = generate_healthcare_data(
        n_samples=1000,
        drift_factor=0.0,
        missing_rate=0.0,
        random_seed=42
    )
    reference_path = output_path / "reference_data.csv"
    reference_data.to_csv(reference_path, index=False)
    logger.info(f"Reference data saved to {reference_path}")
    
    # Generate current data for each scenario
    for scenario in scenarios:
        logger.info(f"Generating {scenario['name']} scenario")
        current_data = generate_healthcare_data(
            n_samples=500,
            drift_factor=scenario['drift_factor'],
            missing_rate=scenario['missing_rate'],
            random_seed=123
        )
        
        current_path = output_path / f"current_data_{scenario['name']}.csv"
        current_data.to_csv(current_path, index=False)
        logger.info(f"Current data ({scenario['name']}) saved to {current_path}")
    
    logger.info(f"All monitoring datasets saved to {output_path}")
    return output_path

def store_sample_data_in_supabase(
    n_patients: int = 50,
    n_lab_results_per_patient: int = 10,
    n_lifestyle_records_per_patient: int = 5
) -> bool:
    """Generate and store sample healthcare data directly in Supabase
    
    Args:
        n_patients: Number of patients to generate
        n_lab_results_per_patient: Number of lab results per patient
        n_lifestyle_records_per_patient: Number of lifestyle records per patient
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        logger.info(f"Generating sample data for {n_patients} patients")
        
        # Generate patient data
        patients_data = []
        for i in range(n_patients):
            patient_id = f"P{i+1:03d}"
            age = np.random.randint(25, 85)
            gender = np.random.choice(['Male', 'Female'])
            
            patients_data.append({
                'id': patient_id,
                'name': f"Patient {i+1}",
                'age': int(age),
                'gender': gender,
                'admission_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                'height': float(np.random.normal(170, 10)),
                'weight': float(np.random.normal(70, 15)),
                'bmi': float(np.random.normal(25, 4)),
                'ethnicity': np.random.choice(['Caucasian', 'Hispanic', 'African American', 'Asian', 'Other']),
                'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured']),
                'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']),
                'phone': f"+1-555-{np.random.randint(1000, 9999)}",
                'medical_history': [],
                'current_medications': [],
                'risk_factors': []
            })
        
        # Insert patients
        result = supabase.client.table('patients').insert(patients_data).execute()
        logger.info(f"Inserted {len(patients_data)} patients")
        
        # Generate lab results
        lab_results_data = []
        lab_types = ['cholesterol', 'blood_glucose', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                    'heart_rate', 'bmi', 'hba1c', 'ldl', 'hdl', 'triglycerides']
        
        for patient_data in patients_data:
            patient_id = patient_data['id']
            for _ in range(n_lab_results_per_patient):
                lab_type = np.random.choice(lab_types)
                
                # Generate realistic values based on lab type
                if lab_type == 'cholesterol':
                    value = np.random.normal(200, 40)
                elif lab_type == 'blood_glucose':
                    value = np.random.normal(100, 20)
                elif lab_type == 'blood_pressure_systolic':
                    value = np.random.normal(120, 20)
                elif lab_type == 'blood_pressure_diastolic':
                    value = np.random.normal(80, 10)
                elif lab_type == 'heart_rate':
                    value = np.random.normal(70, 15)
                elif lab_type == 'bmi':
                    value = np.random.normal(25, 4)
                elif lab_type == 'hba1c':
                    value = np.random.normal(5.5, 1.0)
                elif lab_type == 'ldl':
                    value = np.random.normal(100, 30)
                elif lab_type == 'hdl':
                    value = np.random.normal(50, 15)
                else:  # triglycerides
                    value = np.random.normal(150, 50)
                
                lab_results_data.append({
                    'patient_id': patient_id,
                    'test_name': lab_type,
                    'value': float(max(0, value)),  # Ensure positive values
                    'unit': 'mg/dL' if lab_type in ['cholesterol', 'blood_glucose', 'ldl', 'hdl', 'triglycerides'] else 
                           'mmHg' if 'blood_pressure' in lab_type else 
                           'bpm' if lab_type == 'heart_rate' else 
                           'kg/m²' if lab_type == 'bmi' else '%',
                    'reference_range': '< 200' if lab_type == 'cholesterol' else 'Normal',
                    'status': np.random.choice(['Normal', 'High', 'Low'], p=[0.7, 0.2, 0.1]),
                    'test_date': (datetime.now() - timedelta(days=np.random.randint(1, 90))).isoformat()
                })
        
        # Insert lab results in batches
        batch_size = 100
        for i in range(0, len(lab_results_data), batch_size):
            batch = lab_results_data[i:i+batch_size]
            supabase.client.table('lab_results').insert(batch).execute()
        
        logger.info(f"Inserted {len(lab_results_data)} lab results")
        
        # Generate lifestyle data
        lifestyle_data = []
        lifestyle_types = ['steps', 'sleep_hours', 'exercise_minutes', 'calories_consumed', 'water_intake']
        
        for patient_data in patients_data:
            patient_id = patient_data['id']
            for _ in range(n_lifestyle_records_per_patient):
                lifestyle_type = np.random.choice(lifestyle_types)
                
                if lifestyle_type == 'steps':
                    value = np.random.normal(8000, 3000)
                elif lifestyle_type == 'sleep_hours':
                    value = np.random.normal(7.5, 1.5)
                elif lifestyle_type == 'exercise_minutes':
                    value = np.random.normal(30, 20)
                elif lifestyle_type == 'calories_consumed':
                    value = np.random.normal(2000, 400)
                else:  # water_intake
                    value = np.random.normal(2.5, 0.8)
                
                lifestyle_data.append({
                    'patient_id': patient_id,
                    'activity_type': lifestyle_type,
                    'value': float(max(0, value)),
                    'unit': 'steps' if lifestyle_type == 'steps' else 
                           'hours' if lifestyle_type == 'sleep_hours' else 
                           'minutes' if lifestyle_type == 'exercise_minutes' else 
                           'calories' if lifestyle_type == 'calories_consumed' else 'liters',
                    'recorded_date': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat()
                })
        
        # Insert lifestyle data in batches
        for i in range(0, len(lifestyle_data), batch_size):
            batch = lifestyle_data[i:i+batch_size]
            supabase.client.table('lifestyle_data').insert(batch).execute()
        
        logger.info(f"Inserted {len(lifestyle_data)} lifestyle records")
        
        # Generate some predictions
        predictions_data = []
        for patient_data in patients_data:
            patient_id = patient_data['id']
            # Generate 1-3 predictions per patient
            for _ in range(np.random.randint(1, 4)):
                risk_probability = np.random.uniform(0.1, 0.9)
                predictions_data.append({
                    'patient_id': patient_id,
                    'prediction': 'high_risk' if risk_probability > 0.6 else 'medium_risk' if risk_probability > 0.3 else 'low_risk',
                    'probability': float(risk_probability),
                    'model_name': 'cardiovascular_risk_model_v1',
                    'timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 7))).isoformat()
                })
        
        # Insert predictions
        supabase.client.table('predictions').insert(predictions_data).execute()
        logger.info(f"Inserted {len(predictions_data)} predictions")
        
        logger.info("Successfully generated and stored sample data in Supabase")
        return True
        
    except Exception as e:
        logger.error(f"Error storing sample data in Supabase: {str(e)}")
        return False

def load_monitoring_data(data_dir: str = "data/monitoring") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current data for monitoring
    
    Args:
        data_dir: Directory containing monitoring data files
    
    Returns:
        Tuple of (reference_data, current_data)
    """
    data_path = Path(data_dir)
    
    # Load reference data
    reference_path = data_path / "reference_data.csv"
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found at {reference_path}")
    
    reference_data = pd.read_csv(reference_path)
    logger.info(f"Loaded reference data: {len(reference_data)} samples")
    
    # Load current data (default to medium drift scenario)
    current_path = data_path / "current_data_medium_drift.csv"
    if not current_path.exists():
        # Fallback to any available current data file
        current_files = list(data_path.glob("current_data_*.csv"))
        if not current_files:
            raise FileNotFoundError(f"No current data files found in {data_path}")
        current_path = current_files[0]
    
    current_data = pd.read_csv(current_path)
    logger.info(f"Loaded current data: {len(current_data)} samples from {current_path.name}")
    
    return reference_data, current_data

if __name__ == "__main__":
    # Generate sample datasets
    output_dir = save_monitoring_datasets()
    
    # Test loading
    ref_data, curr_data = load_monitoring_data()
    
    print(f"Reference data shape: {ref_data.shape}")
    print(f"Current data shape: {curr_data.shape}")
    print(f"\nReference data summary:")
    print(ref_data.describe())
    print(f"\nCurrent data summary:")
    print(curr_data.describe())