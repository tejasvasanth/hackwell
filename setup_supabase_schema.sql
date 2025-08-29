-- Comprehensive Supabase schema setup for Healthcare AI System
-- Run this script in your Supabase SQL editor to create the complete database schema

-- Enable Row Level Security (RLS) for all tables
-- This ensures data security and proper access control

-- 1. PATIENTS TABLE
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) UNIQUE NOT NULL, -- External patient identifier
    name VARCHAR(255) NOT NULL,
    age INTEGER CHECK (age >= 0 AND age <= 150),
    gender VARCHAR(20) CHECK (gender IN ('Male', 'Female', 'Other')),
    date_of_birth DATE,
    smoking BOOLEAN DEFAULT FALSE,
    exercise_frequency VARCHAR(50) CHECK (exercise_frequency IN ('low', 'moderate', 'high')),
    family_history JSONB, -- Store family history as JSON
    medical_conditions JSONB, -- Store existing conditions as JSON
    medications JSONB, -- Store current medications as JSON
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. LAB RESULTS TABLE
CREATE TABLE IF NOT EXISTS lab_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id) ON DELETE CASCADE,
    test_date DATE NOT NULL,
    test_type VARCHAR(100) NOT NULL,
    test_name VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    unit VARCHAR(20),
    reference_range VARCHAR(50),
    status VARCHAR(20) CHECK (status IN ('normal', 'abnormal', 'critical')),
    lab_name VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. LIFESTYLE DATA TABLE
CREATE TABLE IF NOT EXISTS lifestyle_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    steps INTEGER CHECK (steps >= 0),
    heart_rate INTEGER CHECK (heart_rate >= 30 AND heart_rate <= 250),
    sleep_hours FLOAT CHECK (sleep_hours >= 0 AND sleep_hours <= 24),
    exercise_minutes INTEGER CHECK (exercise_minutes >= 0),
    calories_burned INTEGER CHECK (calories_burned >= 0),
    weight FLOAT CHECK (weight > 0),
    blood_pressure_systolic INTEGER CHECK (blood_pressure_systolic >= 60 AND blood_pressure_systolic <= 300),
    blood_pressure_diastolic INTEGER CHECK (blood_pressure_diastolic >= 40 AND blood_pressure_diastolic <= 200),
    stress_level INTEGER CHECK (stress_level >= 1 AND stress_level <= 10),
    mood_score INTEGER CHECK (mood_score >= 1 AND mood_score <= 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. PREDICTIONS TABLE
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id) ON DELETE CASCADE,
    prediction_type VARCHAR(50) NOT NULL, -- 'deterioration_risk', 'readmission_risk', etc.
    prediction_value VARCHAR(50) NOT NULL, -- 'low_risk', 'medium_risk', 'high_risk'
    probability FLOAT CHECK (probability >= 0 AND probability <= 1),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    input_features JSONB, -- Store input features as JSON
    explanation JSONB, -- Store SHAP/LIME explanations as JSON
    risk_factors JSONB, -- Store identified risk factors
    recommendations JSONB, -- Store generated recommendations
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. MODEL PERFORMANCE TABLE
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date TIMESTAMP WITH TIME ZONE NOT NULL,
    dataset_type VARCHAR(50) CHECK (dataset_type IN ('train', 'validation', 'test')),
    metric_name VARCHAR(50) NOT NULL, -- 'auroc', 'auprc', 'accuracy', etc.
    metric_value FLOAT NOT NULL,
    sample_size INTEGER,
    metadata JSONB, -- Store additional metrics and configuration
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. DATA QUALITY MONITORING TABLE
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL, -- 'completeness', 'accuracy', 'consistency', etc.
    metric_value FLOAT NOT NULL,
    threshold_value FLOAT,
    status VARCHAR(20) CHECK (status IN ('pass', 'warning', 'fail')),
    check_date TIMESTAMP WITH TIME ZONE NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. FEATURE STORE TABLE
CREATE TABLE IF NOT EXISTS feature_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id) ON DELETE CASCADE,
    feature_group VARCHAR(100) NOT NULL, -- 'demographics', 'labs', 'lifestyle'
    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT,
    feature_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    lookback_days INTEGER, -- For temporal features
    aggregation_type VARCHAR(50), -- 'mean', 'std', 'max', 'min', 'trend'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. ALERTS TABLE
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- 'high_risk', 'data_anomaly', 'model_drift'
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) CHECK (status IN ('active', 'acknowledged', 'resolved')) DEFAULT 'active',
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- CREATE INDEXES FOR PERFORMANCE
-- Patients indexes
CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_age ON patients(age);
CREATE INDEX IF NOT EXISTS idx_patients_gender ON patients(gender);
CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients(created_at);

-- Lab results indexes
CREATE INDEX IF NOT EXISTS idx_lab_results_patient_id ON lab_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_lab_results_test_date ON lab_results(test_date);
CREATE INDEX IF NOT EXISTS idx_lab_results_test_type ON lab_results(test_type);
CREATE INDEX IF NOT EXISTS idx_lab_results_status ON lab_results(status);

-- Lifestyle data indexes
CREATE INDEX IF NOT EXISTS idx_lifestyle_data_patient_id ON lifestyle_data(patient_id);
CREATE INDEX IF NOT EXISTS idx_lifestyle_data_date ON lifestyle_data(date);

-- Predictions indexes
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_type ON predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);

-- Feature store indexes
CREATE INDEX IF NOT EXISTS idx_feature_store_patient_id ON feature_store(patient_id);
CREATE INDEX IF NOT EXISTS idx_feature_store_feature_group ON feature_store(feature_group);
CREATE INDEX IF NOT EXISTS idx_feature_store_timestamp ON feature_store(feature_timestamp);

-- Alerts indexes
CREATE INDEX IF NOT EXISTS idx_alerts_patient_id ON alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON alerts(triggered_at);

-- CREATE TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
DROP TRIGGER IF EXISTS update_patients_updated_at ON patients;
CREATE TRIGGER update_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_lab_results_updated_at ON lab_results;
CREATE TRIGGER update_lab_results_updated_at
    BEFORE UPDATE ON lab_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_lifestyle_data_updated_at ON lifestyle_data;
CREATE TRIGGER update_lifestyle_data_updated_at
    BEFORE UPDATE ON lifestyle_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_predictions_updated_at ON predictions;
CREATE TRIGGER update_predictions_updated_at
    BEFORE UPDATE ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ENABLE ROW LEVEL SECURITY (RLS)
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE lab_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE lifestyle_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_quality_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_store ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- CREATE RLS POLICIES (Allow all operations for authenticated users)
-- In production, you should create more restrictive policies
CREATE POLICY "Allow all operations for authenticated users" ON patients
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON lab_results
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON lifestyle_data
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON predictions
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON model_performance
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON data_quality_metrics
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON feature_store
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON alerts
    FOR ALL USING (auth.role() = 'authenticated');

-- Display setup completion message
SELECT 'Healthcare AI Database Schema Setup Completed Successfully!' as status;