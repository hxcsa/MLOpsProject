# OncoPredict AI: End-to-End MLOps Pipeline for Cancer Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web_App-black)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue)
![Kubeflow](https://img.shields.io/badge/Kubeflow-Orchestration-blue)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)

## 📌 Project Overview
OncoPredict AI is an end-to-end Machine Learning Operations (MLOps) project designed to predict cancer survival outcomes based on patient medical records. This project demonstrates the full lifecycle of a machine learning model, from data preprocessing and feature selection to orchestration, experiment tracking, containerization, and web deployment.

## 🚀 Key Features & Architecture
- **Automated Data Pipeline:** Dynamically processes raw medical data, handles categorical label encoding, and automatically selects the top 10 most impactful predictive features using `SelectKBest` (Chi-Squared).
- **Experiment Tracking:** Integrated with **MLflow** (hosted on DagsHub) to track hyperparameter tuning, model performance metrics (F1 Score, ROC-AUC, Recall), and model versioning.
- **Pipeline Orchestration:** Built a DAG-based orchestration pipeline using **Kubeflow Pipelines (KFP)** to automate the execution of data processing and model training steps in isolated environments.
- **Containerization:** Fully Dockerized the application (`python:3.10-slim`) to ensure reproducibility across any environment.
- **Interactive UI:** Developed a sleek, minimalist web interface using **Flask**, vanilla CSS, and Jinja2 templating, allowing users to input patient vitals and receive real-time predictive confidence scores.

## 🛠️ Tech Stack
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **MLOps & Tracking:** MLflow, DagsHub
* **Orchestration:** Kubeflow Pipelines (KFP SDK v1)
* **Backend & Serving:** Flask
* **Frontend:** HTML5, CSS3 (Modern Minimalist UI)
* **DevOps:** Docker

## 📂 Project Structure
```text
├── artifacts/             # Serialized models, scalers, and processed data
├── kubeflow_pipeline/     # Kubeflow pipeline definitions and compiled YAMLs
├── src/
│   ├── data_proccessing.py # Data ingestion, cleaning, and feature selection
│   ├── model_training.py   # Model training and MLflow integration
│   ├── logger.py           # Custom logging configuration
│   └── custom_exception.py # Custom error handling
├── templates/             # Flask HTML templates
├── static/                # CSS and static assets
├── application.py         # Flask web server and prediction endpoint
├── dockerfile             # Docker container definition
└── requirements.txt       # Project dependencies
```

## 💻 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/oncopredict-mlops.git
cd oncopredict-mlops
```

2. **Run via Docker:**
```bash
docker pull hhxcsa/mlopsapp:latest
docker run -p 5000:5000 hhxcsa/mlopsapp:latest
```

3. **Access the Web App:**
Open your browser and navigate to `http://localhost:5000` to interact with the prediction UI.

## 📈 MLflow Integration
This project logs all runs to a remote MLflow tracking server. You can view the metrics (Accuracy, Precision, Recall) and downloaded model artifacts directly from the DagsHub tracking URI.

## ⚙️ Kubeflow Orchestration
The pipeline can be compiled and executed on any Kubernetes cluster running Kubeflow:
```bash
python kubeflow_pipeline/mlops_pipeline.py
```
This generates an `mlops_pipeline.yaml` file which can be uploaded to the Kubeflow Dashboard to trigger an automated, scalable pipeline run.
