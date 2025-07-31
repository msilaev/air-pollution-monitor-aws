## Environment Variables

You must set the `API_BASE_URL` environment variable in your `.env` file or in your shell before running the dashboard or any API client. For example:

```
API_BASE_URL=http://localhost:8000/api/v1
```
For the AWS cloud deployment it is your ALB URL + `/api/v1`. It is fixed in terraform `/infrastructure/terraform/fargate.tf`.

This is required for the dashboard and API to function correctly.
# Air Pollution Prediction App (MLOps, AWS, Terraform)

This project is an end-to-end MLOps pipeline for predicting air pollution indicators in Helsinki, Finland. It automates data collection, model training, deployment, and monitoring using AWS ECS Fargate, S3, RDS, CloudWatch, and more. Infrastructure is managed with Terraform.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Cloud Deployment](#cloud-deployment)
- [Local Development](#local-development)
- [Testing](#testing)
- [Frontend Dashboard](#frontend-dashboard)
- [Resources](#resources)

---

## Project Overview
- Predicts air pollution using real-time open data from Helsinki air quality stations.
- Fully automated: data ingestion, model training, prediction, monitoring, and dashboard.
- Deployed as microservices on AWS ECS Fargate, with S3 for artifacts and RDS for MLflow backend.
- Infrastructure-as-Code with Terraform.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AWS Cloud Infrastructure                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  Internet Gateway                                                            │
│       │                                                                      │
│  ┌────▼─────┐                                                                │
│  │   ALB    │ (Application Load Balancer)                                    │
│  │ Ports:   │ :80, :5000, :8501                                              │
│  └────┬─────┘                                                                │
│       │                                                                      │
│  ┌────▼───────────────────────────────────────────────────────────────────┐  │
│  │                          ECS Cluster                                 │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │  │
│  │ │   API    │ │ Dashboard│ │  MLflow  │ │  Train   │ │  Predict     │ │  │
│  │ │  :8000   │ │  :8501   │ │  :5000   │ │(scheduled│ │ (scheduled)  │ │  │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│  ┌────▼─────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │      VPC Endpoints (S3, ECR,     │   │   CloudWatch Logs (VPC/IGW)    │  │
│  │      CloudWatch Logs)            │   └─────────────────────────────────┘  │
│  └──────────────────────────────────┘                                        │
│       │                                                                      │
│  ┌────▼─────────────────────────────┐                                        │
│  │        EFS (MLflow Backend)      │                                        │
│  └──────────────────────────────────┘                                        │
│       │                                                                      │
│  ┌────▼─────────────────────────────┐                                        │
│  │            S3 Buckets            │                                        │
│  │  • MLflow Artifacts              │                                        │
│  │  • Data Lake                     │                                        │
│  └──────────────────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

- MLflow uses EFS for its backend store and S3 for artifact storage.
- There is no RDS/PostgreSQL in this deployment.

## Project Structure

```plaintext
infrastructure/
  docker/         # Dockerfiles for all services
  terraform/      # Terraform IaC for AWS (ECS, ALB, S3, etc.)
src/
  api/            # FastAPI backend
  data/           # Data ingestion & processing
  frontend/       # Streamlit dashboard
  models/         # ML model, training, prediction
  monitoring/     # Monitoring & metrics
scripts/          # Utility scripts
README.md         # Documentation
requirements.txt  # Python dependencies
```

## Cloud Deployment

### Prerequisites
- AWS account and credentials
- IAM user with policies from `combined_policies_valid.json`
- Docker and Terraform installed

### AWS Credentials
Set your AWS credentials as environment variables or in `~/.aws/credentials`.

### Build & Push Docker Images
```powershell
& "C:\Program Files\Git\bin\bash.exe" -c 'infrastructure/terraform/build_and_push_all.sh'
```

### Deploy Infrastructure
```powershell
cd infrastructure\terraform
terraform init
terraform apply
```

- This creates all AWS resources and launches ECS services: mlflow, api, dashboard, plus scheduled train and predict tasks.
- The application frontend will be available at the DNS output by Terraform.

### Notes on Networking
- ECS tasks that need public internet access (e.g., for external APIs) run in public subnets with public IPs.
- VPC endpoints are used for S3, ECR, and CloudWatch Logs (one subnet per AZ).
- Security groups are configured for least-privilege access.

## Local Development

### 1. Clone the Repository
```bash
git clone <repository-url>
cd air_pollution_aws
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Activate:
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
- Set `USE_S3` in `src/config.py` to `True` or `False` as needed.
- If using S3, create a `.env` file with your AWS and MLflow settings.

### 5. Quick Start
```bash
python -m src.api.app
streamlit run src/frontend/dashboard_simplified.py
mlflow server --backend-store-uri ./mlruns --default-artifact-root s3://<s3_name>/artifacts --host 0.0.0.0 --port 5000
python -m src.models.train
python -m src.models.prediction
```

- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000
- **MLflow UI:** http://localhost:5000

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=src
```

- Run specific tests as needed (see `tests/` directory).
- Tests use mocking to avoid requiring real AWS credentials or MLflow servers.

## Frontend Dashboard

- The Streamlit dashboard visualizes predictions and data.
- Run locally with `streamlit run src/frontend/dashboard_simplified.py`.

## Resources
- [FMI open data](https://en.ilmatieteenlaitos.fi/open-data)
- [FMI WFS guide](https://en.ilmatieteenlaitos.fi/open-data-manual-wfs-examples-and-guidelines)
- [fmiopen interface](https://github.com/pnuu/fmiopendata)

---

For any issues or questions, please open an issue or contact the maintainer.
