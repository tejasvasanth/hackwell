-- Initialize databases for ML Pipeline

-- Create MLflow database
CREATE DATABASE mlflow;
CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Create Prefect database
CREATE DATABASE prefect;
CREATE USER prefect WITH ENCRYPTED PASSWORD 'prefect';
GRANT ALL PRIVILEGES ON DATABASE prefect TO prefect;

-- Create application database
CREATE DATABASE ml_pipeline;
CREATE USER ml_user WITH ENCRYPTED PASSWORD 'ml_password';
GRANT ALL PRIVILEGES ON DATABASE ml_pipeline TO ml_user;

-- Connect to ml_pipeline database and create tables
\c ml_pipeline;

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    model_version VARCHAR(255),
    input_features JSONB,
    prediction FLOAT,
    confidence FLOAT,
    explanation JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(255),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feature store table
CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(255),
    features JSONB NOT NULL,
    feature_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training data table
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255),
    features JSONB NOT NULL,
    target FLOAT,
    split_type VARCHAR(50), -- 'train', 'validation', 'test'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(255) NOT NULL,
    model_stage VARCHAR(50), -- 'staging', 'production', 'archived'
    model_path VARCHAR(500),
    model_metrics JSONB,
    model_params JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(model_name, model_version)
);

-- Pipeline runs table
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(255) NOT NULL,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50), -- 'running', 'completed', 'failed'
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    parameters JSONB,
    results JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data quality metrics table
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT,
    threshold_value FLOAT,
    status VARCHAR(50), -- 'pass', 'fail', 'warning'
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model monitoring table
CREATE TABLE IF NOT EXISTS model_monitoring (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(255),
    drift_score FLOAT,
    performance_score FLOAT,
    data_quality_score FLOAT,
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_model_name ON predictions(model_name);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX idx_model_metrics_created_at ON model_metrics(created_at);
CREATE INDEX idx_feature_store_entity_id ON feature_store(entity_id);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_created_at ON pipeline_runs(created_at);

-- Grant permissions to ml_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ml_user;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMIT;