from config.supabase_client import SupabaseClient
from datetime import datetime

# Initialize client
client = SupabaseClient()

# Fetch patient data (demographics, lifestyle, labs)
patient = client.get_patient_data(patient_id=1)
print(f"Patient: {patient['name']}, Age: {patient['age']}, BP: {patient['bp']}")

# Upload ECG file to storage
ecg_url = client.upload_ecg(patient_id=1, file_path="path/to/ecg_scan.pdf")
print(f"ECG uploaded: {ecg_url}")

# Log prediction results
result = client.log_prediction(
    patient_id=1, 
    prediction="high_risk", 
    probability=0.85, 
    timestamp=datetime.now()
)
print(f"Prediction logged: {result}")