-- Healthcare Feature Store Tables for Feast + Supabase

-- Demographics Features Table
CREATE TABLE IF NOT EXISTS demographics_features (
    patient_id INTEGER NOT NULL,
    age INTEGER,
    sex VARCHAR(10) CHECK (sex IN ('M', 'F', 'Other')),
    bmi FLOAT CHECK (bmi > 0 AND bmi < 100),
    height FLOAT CHECK (height > 0 AND height < 300), -- cm
    weight FLOAT CHECK (weight > 0 AND weight < 500), -- kg
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (patient_id, event_timestamp)
);

-- Lab Results Features Table
CREATE TABLE IF NOT EXISTS lab_features (
    patient_id INTEGER NOT NULL,
    cholesterol_total FLOAT CHECK (cholesterol_total >= 0),
    cholesterol_ldl FLOAT CHECK (cholesterol_ldl >= 0),
    cholesterol_hdl FLOAT CHECK (cholesterol_hdl >= 0),
    hba1c FLOAT CHECK (hba1c >= 0 AND hba1c <= 20),
    glucose_fasting FLOAT CHECK (glucose_fasting >= 0),
    systolic_bp INTEGER CHECK (systolic_bp > 0 AND systolic_bp < 300),
    diastolic_bp INTEGER CHECK (diastolic_bp > 0 AND diastolic_bp < 200),
    creatinine FLOAT CHECK (creatinine >= 0),
    bun FLOAT CHECK (bun >= 0),
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (patient_id, event_timestamp)
);

-- Lifestyle Features Table
CREATE TABLE IF NOT EXISTS lifestyle_features (
    patient_id INTEGER NOT NULL,
    smoking_status VARCHAR(20) CHECK (smoking_status IN ('never', 'former', 'current')),
    steps_per_day INTEGER CHECK (steps_per_day >= 0),
    exercise_minutes_per_week INTEGER CHECK (exercise_minutes_per_week >= 0),
    alcohol_drinks_per_week INTEGER CHECK (alcohol_drinks_per_week >= 0),
    sleep_hours_per_night FLOAT CHECK (sleep_hours_per_night >= 0 AND sleep_hours_per_night <= 24),
    stress_level INTEGER CHECK (stress_level >= 1 AND stress_level <= 10),
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (patient_id, event_timestamp)
);

-- Wearables Features Table
CREATE TABLE IF NOT EXISTS wearables_features (
    patient_id INTEGER NOT NULL,
    hrv_rmssd FLOAT CHECK (hrv_rmssd >= 0),
    hrv_sdnn FLOAT CHECK (hrv_sdnn >= 0),
    resting_heart_rate INTEGER CHECK (resting_heart_rate > 0 AND resting_heart_rate < 200),
    max_heart_rate INTEGER CHECK (max_heart_rate > 0 AND max_heart_rate < 250),
    ecg_qt_interval FLOAT CHECK (ecg_qt_interval >= 0),
    ecg_pr_interval FLOAT CHECK (ecg_pr_interval >= 0),
    ecg_qrs_duration FLOAT CHECK (ecg_qrs_duration >= 0),
    sleep_efficiency FLOAT CHECK (sleep_efficiency >= 0 AND sleep_efficiency <= 100),
    deep_sleep_minutes INTEGER CHECK (deep_sleep_minutes >= 0),
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (patient_id, event_timestamp)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_demographics_patient_time ON demographics_features(patient_id, event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_labs_patient_time ON lab_features(patient_id, event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lifestyle_patient_time ON lifestyle_features(patient_id, event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_wearables_patient_time ON wearables_features(patient_id, event_timestamp DESC);

-- Insert sample data for testing
INSERT INTO demographics_features (patient_id, age, sex, bmi, height, weight) VALUES
(1, 45, 'M', 26.5, 175, 81.2),
(2, 38, 'F', 22.1, 162, 58.0),
(3, 52, 'M', 31.2, 180, 101.0),
(4, 29, 'F', 19.8, 168, 55.8),
(5, 61, 'M', 28.9, 172, 85.5)
ON CONFLICT (patient_id, event_timestamp) DO NOTHING;

INSERT INTO lab_features (patient_id, cholesterol_total, cholesterol_ldl, cholesterol_hdl, hba1c, glucose_fasting, systolic_bp, diastolic_bp, creatinine, bun) VALUES
(1, 220.5, 140.2, 45.8, 6.2, 105.0, 140, 90, 1.1, 18.5),
(2, 180.2, 110.5, 62.1, 5.4, 88.0, 120, 80, 0.9, 15.2),
(3, 280.8, 180.3, 38.2, 7.8, 145.0, 160, 95, 1.3, 22.1),
(4, 165.3, 95.8, 68.9, 5.1, 82.0, 110, 70, 0.8, 14.0),
(5, 195.7, 125.4, 52.3, 6.0, 98.0, 150, 85, 1.0, 17.8)
ON CONFLICT (patient_id, event_timestamp) DO NOTHING;

INSERT INTO lifestyle_features (patient_id, smoking_status, steps_per_day, exercise_minutes_per_week, alcohol_drinks_per_week, sleep_hours_per_night, stress_level) VALUES
(1, 'current', 6500, 120, 7, 6.5, 7),
(2, 'never', 12000, 300, 2, 8.0, 3),
(3, 'former', 4200, 60, 14, 5.5, 8),
(4, 'never', 15000, 450, 0, 8.5, 2),
(5, 'former', 8500, 180, 5, 7.0, 5)
ON CONFLICT (patient_id, event_timestamp) DO NOTHING;

INSERT INTO wearables_features (patient_id, hrv_rmssd, hrv_sdnn, resting_heart_rate, max_heart_rate, ecg_qt_interval, ecg_pr_interval, ecg_qrs_duration, sleep_efficiency, deep_sleep_minutes) VALUES
(1, 25.4, 45.2, 72, 185, 420.5, 160.2, 95.8, 78.5, 85),
(2, 42.1, 68.9, 58, 195, 380.2, 145.8, 88.2, 92.1, 125),
(3, 18.7, 32.1, 85, 170, 450.8, 180.5, 105.2, 65.8, 45),
(4, 48.9, 75.2, 52, 200, 365.1, 140.2, 82.5, 95.8, 140),
(5, 35.2, 58.4, 68, 175, 410.2, 155.8, 92.1, 82.5, 95)
ON CONFLICT (patient_id, event_timestamp) DO NOTHING;

SELECT 'Healthcare feature tables created successfully!' as status;