import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
from loguru import logger

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