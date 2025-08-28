# Cardiovascular Risk Prediction Training Pipeline

This directory contains a comprehensive training pipeline for cardiovascular risk prediction that integrates **Feast feature store**, **XGBoost modeling**, and **MLflow tracking**.

## 🏗️ Architecture Overview

The training pipeline follows this workflow:

1. **Feature Extraction** → Pull features from Feast feature store
2. **Data Preparation** → Clean and prepare training data
3. **Model Training** → Train XGBoost classifier with cross-validation
4. **Evaluation** → Calculate comprehensive metrics and confusion matrix
5. **Logging** → Log all metrics, parameters, and artifacts to MLflow
6. **Registration** → Register best model to MLflow Model Registry

## 📁 Files Overview

### Core Training Files
- `ml/train_cardiovascular_model.py` - Main training pipeline implementation
- `run_training.py` - Simple script to execute the training pipeline
- `setup_mlflow.py` - MLflow setup and configuration script

### Configuration Files
- `config/settings.py` - Application settings and configuration
- `config/feast_config.py` - Feast feature store configuration
- `config/supabase_client.py` - Supabase database client

## 🚀 Quick Start

### 1. Setup MLflow

```bash
# Setup MLflow experiments and tracking
python setup_mlflow.py
```

### 2. Run Training Pipeline

```bash
# Execute the complete training pipeline
python run_training.py
```

### 3. View Results in MLflow UI

```bash
# Start MLflow UI (in separate terminal)
mlflow ui

# Open browser to: http://localhost:5000
```

## 📊 Features Used

The model uses the following feature categories from Feast:

### Demographics Features
- `demographics:age` - Patient age
- `demographics:gender_encoded` - Gender (0=Female, 1=Male)
- `demographics:bmi` - Body Mass Index

### Lifestyle Features
- `lifestyle:smoking_encoded` - Smoking status (0=No, 1=Yes)
- `lifestyle:exercise_encoded` - Exercise level (0=None, 1=Occasional, 2=Regular)
- `lifestyle:steps_daily` - Average daily steps
- `lifestyle:sleep_hours` - Average sleep hours
- `lifestyle:heart_rate_avg` - Average heart rate
- `lifestyle:activity_level` - Activity level (1-5 scale)
- `lifestyle:stress_level` - Stress level (1-5 scale)

### Lab Results Features
- `labs:bp_systolic` - Systolic blood pressure
- `labs:bp_diastolic` - Diastolic blood pressure
- `labs:cholesterol_total` - Total cholesterol
- `labs:cholesterol_ldl` - LDL cholesterol
- `labs:cholesterol_hdl` - HDL cholesterol
- `labs:glucose_fasting` - Fasting glucose
- `labs:hba1c` - HbA1c levels

## 📈 Metrics Logged to MLflow

### Classification Metrics
- **Accuracy** - Overall prediction accuracy
- **Precision** - Positive predictive value
- **Recall** - Sensitivity/True positive rate
- **F1-Score** - Harmonic mean of precision and recall
- **AUC** - Area under the ROC curve

### Confusion Matrix Components
- **True Positives (TP)** - Correctly predicted high-risk patients
- **False Positives (FP)** - Incorrectly predicted high-risk patients
- **True Negatives (TN)** - Correctly predicted low-risk patients
- **False Negatives (FN)** - Incorrectly predicted low-risk patients

### Cross-Validation Metrics
- **CV Accuracy Mean** - Average accuracy across folds
- **CV Accuracy Std** - Standard deviation of accuracy

### Model Parameters
- All XGBoost hyperparameters
- Dataset information (size, features, splits)
- Training configuration

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=cardiovascular_risk_prediction

# Supabase Configuration (if using real data)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Feast Configuration
FEAST_REPO_PATH=./feast_repo
```

### Model Hyperparameters

Default XGBoost parameters (configurable in `train_cardiovascular_model.py`):

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 'auto',  # Handles class imbalance
    'random_state': 42
}
```

## 🎯 Model Registry

The trained model is automatically registered to MLflow Model Registry with:

- **Model Name**: `cardiovascular_risk_model`
- **Stage**: `Staging` (automatically transitioned)
- **Description**: Comprehensive model metadata
- **Signature**: Input/output schema for deployment
- **Example Input**: Sample data for testing

## 📋 Advanced Usage

### Custom Training Parameters

```python
from ml.train_cardiovascular_model import CardiovascularRiskTrainer

# Initialize trainer with custom experiment
trainer = CardiovascularRiskTrainer(
    experiment_name="custom_cardiovascular_experiment"
)

# Pull features for specific patients
training_data = trainer.pull_features_from_feast(
    entity_ids=["patient_1", "patient_2"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Train with custom parameters
X, y, feature_names = trainer.prepare_training_data(training_data)
results = trainer.train_model(X, y, test_size=0.3, random_state=123)

# Register model with custom name
model_version = trainer.register_best_model(
    results['run_id'], 
    model_name="custom_cardio_model"
)
```

### Batch Prediction

```python
import mlflow.pyfunc

# Load registered model
model = mlflow.pyfunc.load_model(
    model_uri="models:/cardiovascular_risk_model/Staging"
)

# Make predictions
predictions = model.predict(new_patient_data)
```

## 🔍 Monitoring and Debugging

### Logging

All training steps are logged with structured logging:

```bash
# View logs during training
python run_training.py 2>&1 | tee training.log
```

### MLflow UI Navigation

1. **Experiments** → View all training runs
2. **Models** → Browse registered models
3. **Runs** → Detailed metrics and artifacts
4. **Compare** → Compare multiple training runs

### Feature Importance Analysis

Feature importance is automatically logged to MLflow as `feature_importance.txt`:

```
feature                    importance
labs__cholesterol_total    45.2
demographics__age          38.7
labs__bp_systolic          32.1
...
```

## 🚨 Troubleshooting

### Common Issues

1. **MLflow Connection Error**
   ```bash
   # Start MLflow server
   mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **Feast Feature Store Not Found**
   ```bash
   # Initialize Feast repository
   feast init feast_repo
   cd feast_repo
   feast apply
   ```

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Supabase Connection Issues**
   - Check `.env` file configuration
   - Verify Supabase URL and API key
   - Test connection with `test_supabase_client.py`

### Performance Optimization

- **Large Datasets**: Use data sampling or distributed training
- **Feature Selection**: Implement feature selection based on importance scores
- **Hyperparameter Tuning**: Use MLflow with Optuna or Hyperopt
- **Model Serving**: Deploy via MLflow Model Serving or containerization

## 📚 Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Feast Documentation](https://docs.feast.dev/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

## 🤝 Contributing

To extend the training pipeline:

1. **Add New Features**: Update Feast feature definitions
2. **New Models**: Implement in `ml/` directory
3. **Custom Metrics**: Extend `_calculate_metrics()` method
4. **Hyperparameter Tuning**: Add optimization loops
5. **Model Comparison**: Create comparison experiments

---

**Happy Training! 🚀**