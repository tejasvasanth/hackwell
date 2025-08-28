# ML Pipeline Project

A comprehensive machine learning pipeline built with modern MLOps practices, featuring automated training, monitoring, and deployment capabilities.

## 🚀 Features

- **FastAPI** - High-performance REST API for model serving
- **Prefect** - Workflow orchestration and pipeline management
- **XGBoost** - Gradient boosting machine learning framework
- **MLflow** - Experiment tracking and model registry
- **Feast** - Feature store for ML feature management
- **SHAP + LIME** - Model explainability and interpretability
- **Streamlit** - Interactive dashboard for monitoring
- **Supabase** - PostgreSQL database and file storage
- **Docker** - Containerized deployment
- **GitHub Actions** - CI/CD pipeline automation

## 📁 Project Structure

```
├── api/                    # FastAPI application
│   ├── __init__.py
│   └── main.py            # API endpoints and configuration
├── pipelines/             # Prefect workflows
│   ├── __init__.py
│   └── ml_pipeline.py     # ML training and monitoring pipelines
├── ml/                    # Machine learning components
│   ├── __init__.py
│   ├── model.py          # XGBoost model with MLflow integration
│   └── explainer.py      # SHAP/LIME explainability
├── dashboard/             # Streamlit dashboard
│   ├── __init__.py
│   └── app.py            # Interactive monitoring dashboard
├── config/                # Configuration files
│   ├── __init__.py
│   ├── settings.py       # Application settings
│   ├── supabase_client.py # Database client
│   └── feast_config.py   # Feature store configuration
├── tests/                 # Test suite
│   └── __init__.py
├── docker/                # Docker configuration
│   ├── start.sh          # Startup script
│   ├── init-db.sql       # Database initialization
│   └── nginx.conf        # Reverse proxy configuration
├── .github/workflows/     # CI/CD pipelines
│   ├── ci-cd.yml         # Main CI/CD workflow
│   └── model-training.yml # Automated model training
├── data/                  # Data directory (created at runtime)
├── models/                # Model artifacts (created at runtime)
├── logs/                  # Application logs (created at runtime)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Service orchestration
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or 3.11
- Docker and Docker Compose
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hackwell
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Create required directories**
   ```bash
   mkdir -p data models logs mlflow-artifacts feast_repo
   ```

### Docker Setup

1. **Build and start services**
   ```bash
   docker-compose up --build
   ```

2. **Access services**
   - API: http://localhost:8000
   - Dashboard: http://localhost:8501
   - MLflow: http://localhost:5000
   - Prefect UI: http://localhost:4200

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Application Settings
DEBUG=true
API_PORT=8000
STREAMLIT_PORT=8501

# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow

# Prefect Configuration
PREFECT_API_URL=http://localhost:4200/api
PREFECT_WORKSPACE=default

# Feature Store
FEAST_REPO_PATH=./feast_repo

# Model Registry
MODEL_REGISTRY_PATH=./models
```

## 🚀 Usage

### Running Individual Services

1. **Start the API server**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Launch the dashboard**
   ```bash
   streamlit run dashboard/app.py --server.port 8501
   ```

3. **Start MLflow server**
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

4. **Run Prefect server**
   ```bash
   prefect server start --host 0.0.0.0
   ```

### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /models` - List available models
- `POST /retrain` - Trigger model retraining
- `GET /docs` - Interactive API documentation

### Running Pipelines

1. **Training Pipeline**
   ```bash
   python -c "from pipelines.ml_pipeline import ml_training_flow; ml_training_flow()"
   ```

2. **Data Ingestion Pipeline**
   ```bash
   python -c "from pipelines.ml_pipeline import data_ingestion_flow; data_ingestion_flow()"
   ```

3. **Model Monitoring Pipeline**
   ```bash
   python -c "from pipelines.ml_pipeline import model_monitoring_flow; model_monitoring_flow()"
   ```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy api/ ml/ pipelines/
```

## 📊 Monitoring and Observability

### Dashboard Features

- **Overview**: System health and key metrics
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Predictions**: Real-time prediction monitoring
- **Feature Analysis**: Feature importance and distributions
- **Pipeline Status**: Workflow execution status

### MLflow Tracking

- Experiment tracking
- Model versioning
- Parameter and metric logging
- Model registry management

### Logging

Logs are stored in the `logs/` directory:
- `api.log` - API server logs
- `pipeline.log` - Pipeline execution logs
- `model.log` - Model training and inference logs

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **CI/CD Pipeline** (`.github/workflows/ci-cd.yml`)
   - Code quality checks (black, flake8, mypy)
   - Unit tests with coverage
   - Security scanning
   - Docker image building
   - Deployment to staging/production

2. **Model Training Pipeline** (`.github/workflows/model-training.yml`)
   - Scheduled model retraining
   - Data validation
   - Model comparison and selection
   - Automated deployment

### Deployment

#### Staging
- Triggered on push to `develop` branch
- Automated testing and validation
- Manual approval required

#### Production
- Triggered on push to `main` branch
- Full test suite execution
- Model validation
- Blue-green deployment

## 🐳 Docker Services

### Available Services

- **api**: FastAPI application server
- **dashboard**: Streamlit dashboard
- **mlflow**: MLflow tracking server
- **prefect-server**: Prefect orchestration server
- **prefect-worker**: Prefect worker for task execution
- **postgres**: PostgreSQL database
- **redis**: Redis cache
- **nginx**: Reverse proxy and load balancer
- **jupyter**: Jupyter notebook (development profile)

### Service URLs

- Main Application: http://localhost
- API: http://api.localhost
- Dashboard: http://dashboard.localhost
- MLflow: http://mlflow.localhost
- Prefect: http://prefect.localhost

## 🔒 Security

### Best Practices Implemented

- Environment variable configuration
- Database connection pooling
- API rate limiting
- Input validation and sanitization
- CORS configuration
- Security headers
- Container security scanning

### Security Scanning

```bash
# Run security scan
safety check

# Docker image scanning
docker run --rm -v $(pwd):/app aquasec/trivy fs /app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages
- Ensure all CI checks pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Kill process using port
   kill -9 $(lsof -t -i:8000)
   ```

2. **Database connection issues**
   ```bash
   # Check database status
   docker-compose ps postgres
   
   # View database logs
   docker-compose logs postgres
   ```

3. **Model loading errors**
   ```bash
   # Check model directory
   ls -la models/
   
   # Verify MLflow connection
   python -c "import mlflow; print(mlflow.get_tracking_uri())"
   ```

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review the [Documentation](https://your-docs-url.com)
- Contact the development team

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- Prefect team for workflow orchestration
- MLflow team for experiment tracking
- XGBoost team for the ML framework
- Streamlit team for the dashboard framework
- All open-source contributors

---

**Happy Machine Learning! 🤖✨**