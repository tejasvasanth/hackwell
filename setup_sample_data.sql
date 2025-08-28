-- Sample data setup script for Supabase database
-- Run this script in your Supabase SQL editor to create test data

-- Create patients table if it doesn't exist
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    smoking BOOLEAN DEFAULT FALSE,
    exercise VARCHAR(50),
    bp VARCHAR(20),
    cholesterol FLOAT,
    diabetes_status VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create predictions table if it doesn't exist
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    prediction VARCHAR(255) NOT NULL,
    probability FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model_name VARCHAR(255),
    model_version VARCHAR(255),
    input_features JSONB,
    explanation JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert sample patient data
INSERT INTO patients (name, age, gender, smoking, exercise, bp, cholesterol, diabetes_status) VALUES
('John Smith', 45, 'Male', true, 'moderate', '140/90', 220.5, 'pre-diabetic'),
('Sarah Johnson', 38, 'Female', false, 'high', '120/80', 180.2, 'normal'),
('Michael Brown', 52, 'Male', true, 'low', '160/95', 280.8, 'diabetic'),
('Emily Davis', 29, 'Female', false, 'high', '110/70', 165.3, 'normal'),
('Robert Wilson', 61, 'Male', false, 'moderate', '150/85', 195.7, 'pre-diabetic'),
('Lisa Anderson', 34, 'Female', false, 'moderate', '125/75', 175.4, 'normal'),
('David Martinez', 48, 'Male', true, 'low', '170/100', 310.2, 'diabetic'),
('Jennifer Taylor', 42, 'Female', false, 'high', '115/75', 158.9, 'normal'),
('Christopher Lee', 55, 'Male', true, 'moderate', '145/88', 245.6, 'pre-diabetic'),
('Amanda White', 36, 'Female', false, 'high', '118/72', 162.1, 'normal')
ON CONFLICT (id) DO NOTHING;

-- Insert sample prediction data
INSERT INTO predictions (patient_id, prediction, probability, timestamp, model_name, model_version) VALUES
(1, 'high_risk', 0.85, NOW() - INTERVAL '1 day', 'xgboost_v1', '1.0.0'),
(2, 'low_risk', 0.15, NOW() - INTERVAL '2 days', 'xgboost_v1', '1.0.0'),
(3, 'high_risk', 0.92, NOW() - INTERVAL '3 days', 'xgboost_v1', '1.0.0'),
(4, 'low_risk', 0.08, NOW() - INTERVAL '4 days', 'xgboost_v1', '1.0.0'),
(5, 'medium_risk', 0.65, NOW() - INTERVAL '5 days', 'xgboost_v1', '1.0.0'),
(1, 'high_risk', 0.88, NOW() - INTERVAL '6 hours', 'xgboost_v1', '1.0.0'),
(2, 'low_risk', 0.12, NOW() - INTERVAL '12 hours', 'xgboost_v1', '1.0.0'),
(6, 'low_risk', 0.18, NOW() - INTERVAL '18 hours', 'xgboost_v1', '1.0.0'),
(7, 'high_risk', 0.94, NOW() - INTERVAL '1 day 6 hours', 'xgboost_v1', '1.0.0'),
(8, 'low_risk', 0.09, NOW() - INTERVAL '2 days 12 hours', 'xgboost_v1', '1.0.0')
ON CONFLICT (id) DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_patients_age ON patients(age);
CREATE INDEX IF NOT EXISTS idx_patients_gender ON patients(gender);
CREATE INDEX IF NOT EXISTS idx_patients_diabetes_status ON patients(diabetes_status);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);

-- Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to patients table
DROP TRIGGER IF EXISTS update_patients_updated_at ON patients;
CREATE TRIGGER update_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply the trigger to predictions table
DROP TRIGGER IF EXISTS update_predictions_updated_at ON predictions;
CREATE TRIGGER update_predictions_updated_at
    BEFORE UPDATE ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Display summary of inserted data
SELECT 'Sample data setup completed!' as status;
SELECT COUNT(*) as total_patients FROM patients;
SELECT COUNT(*) as total_predictions FROM predictions;

-- Show sample patient data
SELECT 
    id, 
    name, 
    age, 
    gender, 
    smoking, 
    exercise, 
    bp, 
    cholesterol, 
    diabetes_status 
FROM patients 
LIMIT 5;

-- Show sample prediction data
SELECT 
    p.id,
    pt.name as patient_name,
    p.prediction,
    p.probability,
    p.timestamp,
    p.model_name
FROM predictions p
JOIN patients pt ON p.patient_id = pt.id
ORDER BY p.timestamp DESC
LIMIT 5;