-- Sample data insertion for Healthcare AI System
-- Run this script after setting up the schema to populate tables with realistic data

-- Insert sample patients
INSERT INTO patients (patient_id, name, age, gender, date_of_birth, smoking, exercise_frequency, family_history, medical_conditions, medications) VALUES
('P001', 'John Smith', 65, 'Male', '1959-03-15', true, 'low', 
 '{"diabetes": true, "heart_disease": true, "hypertension": false}',
 '{"diabetes_type2": {"diagnosed": "2018-05-12", "severity": "moderate"}, "hypertension": {"diagnosed": "2020-01-08", "severity": "mild"}}',
 '{"metformin": {"dosage": "500mg", "frequency": "twice_daily"}, "lisinopril": {"dosage": "10mg", "frequency": "once_daily"}}'),

('P002', 'Sarah Johnson', 45, 'Female', '1979-07-22', false, 'moderate',
 '{"diabetes": false, "heart_disease": false, "hypertension": true}',
 '{"hypothyroidism": {"diagnosed": "2019-03-20", "severity": "mild"}}',
 '{"levothyroxine": {"dosage": "75mcg", "frequency": "once_daily"}}'),

('P003', 'Michael Brown', 72, 'Male', '1952-11-08', false, 'high',
 '{"diabetes": true, "heart_disease": true, "hypertension": true}',
 '{"coronary_artery_disease": {"diagnosed": "2015-09-14", "severity": "moderate"}, "diabetes_type2": {"diagnosed": "2010-02-28", "severity": "well_controlled"}}',
 '{"atorvastatin": {"dosage": "40mg", "frequency": "once_daily"}, "metformin": {"dosage": "1000mg", "frequency": "twice_daily"}, "aspirin": {"dosage": "81mg", "frequency": "once_daily"}}'),

('P004', 'Emily Davis', 38, 'Female', '1986-04-12', false, 'high',
 '{"diabetes": false, "heart_disease": false, "hypertension": false}',
 '{}', '{}'),

('P005', 'Robert Wilson', 58, 'Male', '1966-09-30', true, 'low',
 '{"diabetes": false, "heart_disease": true, "hypertension": true}',
 '{"hypertension": {"diagnosed": "2017-06-15", "severity": "moderate"}, "copd": {"diagnosed": "2021-11-03", "severity": "mild"}}',
 '{"amlodipine": {"dosage": "5mg", "frequency": "once_daily"}, "albuterol": {"dosage": "90mcg", "frequency": "as_needed"}}'),

('P006', 'Lisa Anderson', 52, 'Female', '1972-01-18', false, 'moderate',
 '{"diabetes": true, "heart_disease": false, "hypertension": false}',
 '{"diabetes_type2": {"diagnosed": "2022-08-10", "severity": "newly_diagnosed"}}',
 '{"metformin": {"dosage": "500mg", "frequency": "once_daily"}}'),

('P007', 'David Martinez', 67, 'Male', '1957-12-05', false, 'moderate',
 '{"diabetes": false, "heart_disease": true, "hypertension": true}',
 '{"atrial_fibrillation": {"diagnosed": "2020-04-22", "severity": "controlled"}, "hypertension": {"diagnosed": "2016-07-11", "severity": "moderate"}}',
 '{"warfarin": {"dosage": "5mg", "frequency": "once_daily"}, "metoprolol": {"dosage": "50mg", "frequency": "twice_daily"}}'),

('P008', 'Jennifer Taylor', 41, 'Female', '1983-05-28', false, 'high',
 '{"diabetes": false, "heart_disease": false, "hypertension": false}',
 '{}', '{}'),

('P009', 'Christopher Lee', 69, 'Male', '1955-08-14', true, 'low',
 '{"diabetes": true, "heart_disease": true, "hypertension": true}',
 '{"diabetes_type2": {"diagnosed": "2012-03-07", "severity": "poorly_controlled"}, "heart_failure": {"diagnosed": "2019-10-15", "severity": "moderate"}}',
 '{"insulin_glargine": {"dosage": "20_units", "frequency": "once_daily"}, "furosemide": {"dosage": "40mg", "frequency": "once_daily"}, "carvedilol": {"dosage": "12.5mg", "frequency": "twice_daily"}}'),

('P010', 'Amanda White', 55, 'Female', '1969-02-09', false, 'moderate',
 '{"diabetes": false, "heart_disease": false, "hypertension": true}',
 '{"hypertension": {"diagnosed": "2018-12-03", "severity": "mild"}, "osteoporosis": {"diagnosed": "2023-01-20", "severity": "mild"}}',
 '{"lisinopril": {"dosage": "5mg", "frequency": "once_daily"}, "alendronate": {"dosage": "70mg", "frequency": "weekly"}}');

-- Insert lab results for the past 6 months
INSERT INTO lab_results (patient_id, test_date, test_type, test_name, value, unit, reference_range, status, lab_name) VALUES
-- Patient P001 (John Smith) - Diabetic with hypertension
('P001', '2024-01-15', 'Blood Chemistry', 'Glucose (Fasting)', 145, 'mg/dL', '70-100', 'abnormal', 'Central Lab'),
('P001', '2024-01-15', 'Blood Chemistry', 'HbA1c', 8.2, '%', '<7.0', 'abnormal', 'Central Lab'),
('P001', '2024-01-15', 'Lipid Panel', 'Total Cholesterol', 220, 'mg/dL', '<200', 'abnormal', 'Central Lab'),
('P001', '2024-01-15', 'Lipid Panel', 'LDL Cholesterol', 145, 'mg/dL', '<100', 'abnormal', 'Central Lab'),
('P001', '2024-01-15', 'Lipid Panel', 'HDL Cholesterol', 38, 'mg/dL', '>40', 'abnormal', 'Central Lab'),
('P001', '2024-01-15', 'Kidney Function', 'Creatinine', 1.3, 'mg/dL', '0.7-1.3', 'normal', 'Central Lab'),
('P001', '2024-01-15', 'Kidney Function', 'eGFR', 58, 'mL/min/1.73m²', '>60', 'abnormal', 'Central Lab'),

-- Patient P002 (Sarah Johnson) - Hypothyroidism
('P002', '2024-01-20', 'Thyroid Function', 'TSH', 2.8, 'mIU/L', '0.4-4.0', 'normal', 'Central Lab'),
('P002', '2024-01-20', 'Thyroid Function', 'Free T4', 1.2, 'ng/dL', '0.8-1.8', 'normal', 'Central Lab'),
('P002', '2024-01-20', 'Blood Chemistry', 'Glucose (Fasting)', 92, 'mg/dL', '70-100', 'normal', 'Central Lab'),
('P002', '2024-01-20', 'Lipid Panel', 'Total Cholesterol', 185, 'mg/dL', '<200', 'normal', 'Central Lab'),

-- Patient P003 (Michael Brown) - CAD and diabetes
('P003', '2024-01-10', 'Blood Chemistry', 'Glucose (Fasting)', 118, 'mg/dL', '70-100', 'abnormal', 'Central Lab'),
('P003', '2024-01-10', 'Blood Chemistry', 'HbA1c', 6.8, '%', '<7.0', 'normal', 'Central Lab'),
('P003', '2024-01-10', 'Cardiac Markers', 'Troponin I', 0.02, 'ng/mL', '<0.04', 'normal', 'Central Lab'),
('P003', '2024-01-10', 'Lipid Panel', 'LDL Cholesterol', 85, 'mg/dL', '<100', 'normal', 'Central Lab'),

-- Patient P009 (Christopher Lee) - Poorly controlled diabetes with heart failure
('P009', '2024-01-25', 'Blood Chemistry', 'Glucose (Fasting)', 195, 'mg/dL', '70-100', 'abnormal', 'Central Lab'),
('P009', '2024-01-25', 'Blood Chemistry', 'HbA1c', 9.5, '%', '<7.0', 'abnormal', 'Central Lab'),
('P009', '2024-01-25', 'Cardiac Markers', 'BNP', 850, 'pg/mL', '<100', 'abnormal', 'Central Lab'),
('P009', '2024-01-25', 'Kidney Function', 'Creatinine', 1.8, 'mg/dL', '0.7-1.3', 'abnormal', 'Central Lab'),
('P009', '2024-01-25', 'Kidney Function', 'eGFR', 38, 'mL/min/1.73m²', '>60', 'abnormal', 'Central Lab');

-- Insert lifestyle data for the past 30 days
INSERT INTO lifestyle_data (patient_id, date, steps, heart_rate, sleep_hours, exercise_minutes, calories_burned, weight, blood_pressure_systolic, blood_pressure_diastolic, stress_level, mood_score) VALUES
-- Patient P001 (High risk - poor lifestyle)
('P001', '2024-01-20', 3200, 78, 6.5, 15, 180, 185.5, 145, 92, 7, 5),
('P001', '2024-01-21', 2800, 82, 6.0, 0, 150, 185.8, 148, 95, 8, 4),
('P001', '2024-01-22', 4100, 75, 7.0, 30, 220, 185.2, 142, 88, 6, 6),

-- Patient P002 (Moderate risk - good lifestyle)
('P002', '2024-01-20', 8500, 68, 7.5, 45, 320, 142.3, 118, 76, 4, 7),
('P002', '2024-01-21', 9200, 72, 8.0, 60, 380, 142.1, 115, 74, 3, 8),
('P002', '2024-01-22', 7800, 70, 7.8, 40, 295, 142.5, 120, 78, 4, 7),

-- Patient P003 (Moderate risk - excellent lifestyle despite age)
('P003', '2024-01-20', 6800, 65, 8.2, 75, 420, 168.7, 128, 82, 3, 8),
('P003', '2024-01-21', 7200, 68, 8.0, 80, 450, 168.5, 125, 80, 2, 9),
('P003', '2024-01-22', 6500, 66, 8.5, 70, 400, 168.9, 130, 84, 3, 8),

-- Patient P004 (Low risk - very active)
('P004', '2024-01-20', 12500, 62, 8.5, 90, 520, 128.4, 108, 68, 2, 9),
('P004', '2024-01-21', 11800, 65, 8.2, 85, 495, 128.2, 110, 70, 2, 9),
('P004', '2024-01-22', 13200, 60, 8.8, 95, 550, 128.6, 106, 66, 1, 10),

-- Patient P009 (Very high risk - poor lifestyle)
('P009', '2024-01-20', 1800, 88, 5.5, 0, 120, 195.8, 165, 98, 9, 3),
('P009', '2024-01-21', 1500, 92, 5.0, 0, 100, 196.2, 170, 102, 9, 2),
('P009', '2024-01-22', 2100, 85, 6.0, 10, 140, 195.5, 162, 95, 8, 4);

-- Insert predictions for all patients
INSERT INTO predictions (patient_id, prediction_type, prediction_value, probability, confidence_score, prediction_date, model_name, model_version, input_features, explanation, risk_factors, recommendations) VALUES
('P001', 'deterioration_risk', 'high_risk', 0.78, 0.85, '2024-01-25 10:30:00', 'XGBoost_Risk_Model', 'v2.1.0',
 '{"age": 65, "smoking": true, "hba1c": 8.2, "egfr": 58, "exercise_freq": "low", "bp_systolic": 145}',
 '{"shap_values": {"hba1c": 0.25, "smoking": 0.18, "age": 0.15, "egfr": 0.12, "exercise_freq": 0.08}}',
 '["Poor glycemic control (HbA1c 8.2%)", "Active smoking", "Reduced kidney function", "Sedentary lifestyle", "Elevated blood pressure"]',
 '["Urgent diabetes management consultation", "Smoking cessation program", "Nephrology referral", "Structured exercise program", "Blood pressure optimization"]'),

('P002', 'deterioration_risk', 'low_risk', 0.15, 0.92, '2024-01-25 10:35:00', 'XGBoost_Risk_Model', 'v2.1.0',
 '{"age": 45, "smoking": false, "thyroid_controlled": true, "exercise_freq": "moderate", "bp_normal": true}',
 '{"shap_values": {"age": -0.20, "smoking": -0.15, "exercise_freq": -0.12, "thyroid_controlled": -0.08}}',
 '["Well-controlled hypothyroidism", "Non-smoker", "Regular exercise", "Normal blood pressure"]',
 '["Continue current thyroid medication", "Maintain exercise routine", "Annual health screening", "Preventive care focus"]'),

('P003', 'deterioration_risk', 'medium_risk', 0.42, 0.88, '2024-01-25 10:40:00', 'XGBoost_Risk_Model', 'v2.1.0',
 '{"age": 72, "cad_history": true, "hba1c": 6.8, "exercise_freq": "high", "medications_adherent": true}',
 '{"shap_values": {"age": 0.18, "cad_history": 0.15, "hba1c": 0.05, "exercise_freq": -0.12, "medications": -0.08}}',
 '["Advanced age", "History of coronary artery disease", "Multiple comorbidities"]',
 '["Continue excellent lifestyle habits", "Regular cardiology follow-up", "Medication adherence monitoring", "Cardiac rehabilitation maintenance"]'),

('P004', 'deterioration_risk', 'low_risk', 0.08, 0.95, '2024-01-25 10:45:00', 'XGBoost_Risk_Model', 'v2.1.0',
 '{"age": 38, "smoking": false, "exercise_freq": "high", "no_comorbidities": true, "excellent_vitals": true}',
 '{"shap_values": {"age": -0.25, "smoking": -0.15, "exercise_freq": -0.18, "comorbidities": -0.10}}',
 '["Young age", "Excellent physical fitness", "No chronic conditions", "Optimal lifestyle habits"]',
 '["Maintain current lifestyle", "Preventive screening as appropriate for age", "Continue regular exercise", "Healthy diet maintenance"]'),

('P009', 'deterioration_risk', 'high_risk', 0.89, 0.91, '2024-01-25 10:50:00', 'XGBoost_Risk_Model', 'v2.1.0',
 '{"age": 69, "smoking": true, "hba1c": 9.5, "heart_failure": true, "egfr": 38, "exercise_freq": "low"}',
 '{"shap_values": {"hba1c": 0.28, "heart_failure": 0.22, "egfr": 0.18, "smoking": 0.15, "age": 0.12}}',
 '["Severely uncontrolled diabetes", "Heart failure", "Chronic kidney disease", "Active smoking", "Advanced age", "Sedentary lifestyle"]',
 '["Immediate endocrinology consultation", "Heart failure management optimization", "Nephrology urgent referral", "Smoking cessation (priority)", "Supervised exercise program", "Frequent monitoring"]');

-- Insert model performance metrics
INSERT INTO model_performance (model_name, model_version, evaluation_date, dataset_type, metric_name, metric_value, sample_size, metadata) VALUES
('XGBoost_Risk_Model', 'v2.1.0', '2024-01-20 14:00:00', 'test', 'auroc', 0.87, 2500, '{"cross_validation": true, "folds": 5}'),
('XGBoost_Risk_Model', 'v2.1.0', '2024-01-20 14:00:00', 'test', 'auprc', 0.82, 2500, '{"cross_validation": true, "folds": 5}'),
('XGBoost_Risk_Model', 'v2.1.0', '2024-01-20 14:00:00', 'test', 'accuracy', 0.79, 2500, '{"threshold": 0.5}'),
('XGBoost_Risk_Model', 'v2.1.0', '2024-01-20 14:00:00', 'test', 'precision', 0.75, 2500, '{"threshold": 0.5}'),
('XGBoost_Risk_Model', 'v2.1.0', '2024-01-20 14:00:00', 'test', 'recall', 0.83, 2500, '{"threshold": 0.5}'),
('XGBoost_Risk_Model', 'v2.1.0', '2024-01-20 14:00:00', 'test', 'f1_score', 0.79, 2500, '{"threshold": 0.5}');

-- Insert data quality metrics
INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, threshold_value, status, check_date, details) VALUES
('patients', 'completeness', 0.98, 0.95, 'pass', '2024-01-25 09:00:00', '{"missing_fields": ["family_history"], "missing_count": 2}'),
('lab_results', 'completeness', 0.99, 0.95, 'pass', '2024-01-25 09:00:00', '{"missing_fields": [], "missing_count": 0}'),
('lifestyle_data', 'completeness', 0.94, 0.95, 'warning', '2024-01-25 09:00:00', '{"missing_fields": ["sleep_hours", "stress_level"], "missing_count": 15}'),
('predictions', 'accuracy', 0.87, 0.80, 'pass', '2024-01-25 09:00:00', '{"validation_method": "cross_validation", "sample_size": 1000}');

-- Insert feature store data (engineered features)
INSERT INTO feature_store (patient_id, feature_group, feature_name, feature_value, feature_timestamp, lookback_days, aggregation_type) VALUES
('P001', 'demographics', 'age_normalized', 0.65, '2024-01-25 10:00:00', NULL, NULL),
('P001', 'labs', 'hba1c_trend_30d', 0.15, '2024-01-25 10:00:00', 30, 'trend'),
('P001', 'labs', 'glucose_mean_7d', 142.5, '2024-01-25 10:00:00', 7, 'mean'),
('P001', 'lifestyle', 'steps_mean_7d', 3366.7, '2024-01-25 10:00:00', 7, 'mean'),
('P001', 'lifestyle', 'exercise_consistency_30d', 0.23, '2024-01-25 10:00:00', 30, 'consistency'),

('P002', 'demographics', 'age_normalized', 0.45, '2024-01-25 10:00:00', NULL, NULL),
('P002', 'labs', 'tsh_stability_90d', 0.92, '2024-01-25 10:00:00', 90, 'stability'),
('P002', 'lifestyle', 'steps_mean_7d', 8500.0, '2024-01-25 10:00:00', 7, 'mean'),
('P002', 'lifestyle', 'exercise_consistency_30d', 0.87, '2024-01-25 10:00:00', 30, 'consistency'),

('P009', 'demographics', 'age_normalized', 0.69, '2024-01-25 10:00:00', NULL, NULL),
('P009', 'labs', 'hba1c_trend_30d', 0.25, '2024-01-25 10:00:00', 30, 'trend'),
('P009', 'labs', 'bnp_trend_30d', 0.18, '2024-01-25 10:00:00', 30, 'trend'),
('P009', 'lifestyle', 'steps_mean_7d', 1800.0, '2024-01-25 10:00:00', 7, 'mean'),
('P009', 'lifestyle', 'exercise_consistency_30d', 0.05, '2024-01-25 10:00:00', 30, 'consistency');

-- Insert alerts
INSERT INTO alerts (patient_id, alert_type, severity, title, description, status, triggered_at, metadata) VALUES
('P001', 'high_risk', 'high', 'High Deterioration Risk Detected', 'Patient shows multiple risk factors including poor glycemic control and smoking', 'active', '2024-01-25 10:30:00', '{"risk_score": 0.78, "primary_factors": ["hba1c", "smoking"]}'),

('P009', 'high_risk', 'critical', 'Critical Risk - Immediate Attention Required', 'Patient has very high deterioration risk with multiple severe comorbidities', 'active', '2024-01-25 10:50:00', '{"risk_score": 0.89, "primary_factors": ["hba1c", "heart_failure", "ckd"]}'),

('P002', 'data_anomaly', 'low', 'Missing Lifestyle Data', 'Some lifestyle metrics missing for the past 3 days', 'acknowledged', '2024-01-23 08:00:00', '{"missing_days": 3, "missing_metrics": ["steps", "sleep_hours"]}'),

('P003', 'model_drift', 'medium', 'Model Performance Monitoring', 'Slight decrease in model accuracy for this patient profile', 'resolved', '2024-01-22 15:30:00', '{"accuracy_drop": 0.03, "patient_profile": "elderly_cad_diabetes"}}');

-- Display completion status
SELECT 'Sample Data Insertion Completed Successfully!' as status;
SELECT COUNT(*) as patient_count FROM patients;
SELECT COUNT(*) as lab_results_count FROM lab_results;
SELECT COUNT(*) as lifestyle_data_count FROM lifestyle_data;
SELECT COUNT(*) as predictions_count FROM predictions;
SELECT COUNT(*) as alerts_count FROM alerts;

-- Show sample data
SELECT 'Sample Patients:' as info;
SELECT patient_id, name, age, gender FROM patients LIMIT 5;

SELECT 'Sample Predictions:' as info;
SELECT patient_id, prediction_type, prediction_value, probability FROM predictions LIMIT 5;