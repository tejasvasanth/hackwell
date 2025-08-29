# Healthcare AI Data Pipeline Documentation

## Overview

This document describes the comprehensive data pipeline implementation for the Healthcare AI system, designed to handle 30-180 days of patient historical data for training deterioration prediction models.

## Pipeline Architecture

### Core Components

1. **Historical Data Extraction** (`extract_historical_patient_data`)
   - Retrieves patient demographics, lab results, and lifestyle data
   - Supports configurable lookback periods (30-180 days)
   - Filters patients with minimum history requirements
   - Integrates with Supabase database

2. **Temporal Feature Engineering**
   - `create_temporal_lab_features`: Time-series features from lab results
   - `create_temporal_lifestyle_features`: Time-series features from lifestyle data
   - Multiple time windows: 7, 30, 90 days, and full lookback period
   - Statistical aggregations: mean, std, min, max, count, trend analysis

3. **Deterioration Label Generation** (`create_deterioration_labels`)
   - Creates 90-day deterioration prediction targets
   - Based on multiple risk factors and temporal trends
   - Supports supervised learning model training

4. **Feature Store Integration**
   - Loads processed features into Feast feature store
   - Categorized feature groups: demographics, lifestyle, labs
   - Enables real-time feature serving for predictions

## Pipeline Flows

### 1. Historical Data Pipeline (`historical_data_pipeline`)

**Purpose**: Main pipeline for processing historical patient data

**Parameters**:
- `lookback_days` (default: 180): Days of historical data to process
- `min_history_days` (default: 30): Minimum history required per patient
- `limit` (default: 1000): Maximum number of patients to process

**Workflow**:
1. Extract historical patient data
2. Create deterioration labels
3. Transform data for ML training
4. Load features to Feast
5. Trigger model training
6. Calculate pipeline statistics

**Schedule**: Weekly (Sunday at midnight)

### 2. Healthcare ETL Pipeline (`healthcare_etl_pipeline`)

**Purpose**: Daily data extraction and transformation

**Workflow**:
1. Extract patient demographics
2. Extract lab results
3. Extract lifestyle/wearable data
4. Aggregate lifestyle data (30-day averages)
5. Transform to ML-ready features
6. Load to Feast feature store

**Schedule**: Daily (1 AM)

### 3. ML Training Pipeline (`ml_training_pipeline`)

**Purpose**: Model training and validation

**Workflow**:
1. Load training data
2. Preprocess data (handle missing values, split datasets)
3. Train ML model with MLflow tracking
4. Validate model performance
5. Generate model explanations

**Schedule**: Daily (2 AM)

## Feature Engineering Details

### Temporal Lab Features

For each lab test (glucose, cholesterol, blood pressure, etc.):
- **7-day window**: Recent short-term trends
- **30-day window**: Medium-term patterns
- **90-day window**: Long-term trends
- **Full lookback**: Complete historical context

**Calculated metrics per window**:
- Mean, standard deviation, min, max, count
- Trend (linear slope)
- Recent vs. historical comparison ratios

### Temporal Lifestyle Features

For lifestyle metrics (steps, heart rate, sleep, exercise):
- Same time windows as lab features
- Additional activity pattern analysis
- Variability and consistency metrics

### Risk Factor Integration

Deterioratoin labels consider:
- Age-based risk factors
- Lab value thresholds (glucose > 140, cholesterol > 240)
- Blood pressure trends
- Activity level decline
- Sleep pattern disruption
- Combined risk scoring (0-1 scale)

## Data Quality and Validation

### Data Validation Steps
1. **Completeness Check**: Minimum required data points per time window
2. **Outlier Detection**: Statistical outlier identification and handling
3. **Missing Value Strategy**: Forward fill, interpolation, or exclusion
4. **Temporal Consistency**: Ensure chronological data ordering

### Quality Metrics
- Data completeness percentage per patient
- Feature coverage across time windows
- Missing value rates by feature type
- Temporal data gaps identification

## Deployment and Monitoring

### Prefect Integration

**Note**: Current implementation uses Prefect 2.x compatible syntax. Deployment configurations have been updated to use modern Prefect CLI commands.

**Deployment Commands**:
```bash
# Deploy historical data pipeline
prefect deploy --name historical-data-weekly

# Deploy healthcare ETL
prefect deploy --name healthcare-etl-daily

# Deploy ML training
prefect deploy --name ml-training-daily
```

### Monitoring Pipeline (`model_monitoring_pipeline`)

**Purpose**: Monitor model performance and data drift

**Workflow**:
1. Check model performance metrics
2. Detect data drift
3. Monitor prediction quality
4. Generate alerts for anomalies

**Schedule**: Every 4 hours

## Testing and Validation

### Pipeline Testing

A comprehensive test suite validates:
- Temporal feature extraction accuracy
- Deterioration risk calculation logic
- Data processing pipeline integrity
- Integration between components

**Test Execution**:
```bash
python test_simple_pipeline.py
```

**Test Coverage**:
- ✓ Temporal feature extraction
- ✓ Risk score calculation
- ✓ Data processing workflows
- ✓ Pipeline component integration

## Performance Considerations

### Scalability
- Batch processing for large patient cohorts
- Configurable limits to manage resource usage
- Parallel processing for independent patient records
- Efficient database queries with proper indexing

### Resource Management
- Memory-efficient data processing
- Incremental feature updates
- Cached intermediate results
- Optimized database connections

## Usage Examples

### Running Historical Data Pipeline

```python
from pipelines.ml_pipeline import historical_data_pipeline

# Process 6 months of data for 500 patients
result = historical_data_pipeline(
    lookback_days=180,
    min_history_days=30,
    limit=500
)
```

### Accessing Pipeline Functions

```python
from pipelines.ml_pipeline import (
    extract_historical_patient_data,
    create_temporal_lab_features,
    create_temporal_lifestyle_features,
    create_deterioration_labels
)

# Extract data for specific patient
patient_data = extract_historical_patient_data(
    lookback_days=90,
    patient_id="P001"
)

# Create temporal features
lab_features = create_temporal_lab_features(patient_data['labs'])
lifestyle_features = create_temporal_lifestyle_features(patient_data['lifestyle'])

# Generate deterioration labels
labels = create_deterioration_labels(patient_data)
```

## Integration with ML System

### Feature Store Integration
- Features automatically loaded to Feast
- Real-time feature serving for predictions
- Historical feature retrieval for training
- Feature versioning and lineage tracking

### Model Training Integration
- Automated model retraining with new data
- MLflow experiment tracking
- Model performance monitoring
- A/B testing support for model versions

### Dashboard Integration
- Real-time pipeline status monitoring
- Data quality metrics visualization
- Model performance tracking
- Alert management interface

## Conclusion

The Healthcare AI data pipeline provides a robust, scalable solution for processing historical patient data and training deterioration prediction models. The implementation includes:

- ✅ **Comprehensive Data Processing**: 30-180 days historical data handling
- ✅ **Advanced Feature Engineering**: Temporal features across multiple time windows
- ✅ **Automated Workflows**: Prefect-based orchestration with scheduling
- ✅ **Quality Assurance**: Extensive testing and validation
- ✅ **Scalable Architecture**: Configurable limits and efficient processing
- ✅ **Integration Ready**: Seamless integration with ML models and dashboard

The pipeline is production-ready and supports the full lifecycle of the Healthcare AI system, from data ingestion to model deployment and monitoring.