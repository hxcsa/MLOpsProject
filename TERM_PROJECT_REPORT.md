# Term Project Report: OncoPredict AI
## End-to-End MLOps Pipeline for Breast Cancer Survival Prediction

**Course:** [Insert Course Name]
**Student:** Alhas
**Date:** May 16, 2026

---

## 1. Abstract
The "OncoPredict AI" project is a comprehensive Machine Learning Operations (MLOps) pipeline designed to predict breast cancer survival outcomes based on patient medical records. Rather than stopping at model creation, this project demonstrates a fully production-ready lifecycle. It encompasses automated data ingestion, preprocessing, dynamic feature selection, robust model training, centralized experiment tracking, scalable orchestration, containerization, and interactive web deployment. The ultimate goal of this project is to bridge the gap between data science experimentation and software engineering production standards.

## 2. Introduction & Objectives
Breast cancer is one of the most prevalent cancers worldwide, and accurately predicting patient outcomes based on clinical data can aid in treatment planning. However, deploying machine learning models in healthcare requires reliability, reproducibility, and scalability.

**Objectives of this Project:**
- Build an accurate machine learning classification model to predict patient survival.
- Automate the data processing and training pipelines to ensure reproducibility.
- Track hyperparameter tuning and model metrics systematically.
- Deploy the model via an interactive web interface.
- Package the entire solution using modern DevOps and MLOps tools.

## 3. System Architecture and Tech Stack
The architecture of this project is modular, consisting of several interdependent components:

- **Data Processing & ML:** Python, Pandas, NumPy, Scikit-Learn
- **Experiment Tracking:** MLflow (hosted remotely via DagsHub)
- **Pipeline Orchestration:** Kubeflow Pipelines (KFP)
- **Containerization:** Docker (`python:3.10-slim`)
- **Backend Serving:** Flask
- **Frontend UI:** HTML5, Vanilla CSS3 (Modern Minimalist Design)

## 4. Data Processing Pipeline (`src/data_proccessing.py`)
The data pipeline is responsible for transforming raw medical data into a format suitable for machine learning models. 
1. **Data Ingestion:** Loads raw patient data (`data.csv`).
2. **Categorical Encoding:** Uses `LabelEncoder` to convert categorical variables (e.g., Treatment Type, Diabetes, Country) into numerical representations.
3. **Feature Selection:** Applies `SelectKBest` with the `chi2` statistical test to dynamically select the top 10 most predictive features (e.g., Healthcare Costs, Tumor Size, Mortality Rate). This reduces dimensionality and prevents overfitting.
4. **Data Splitting & Scaling:** Splits the data into an 80/20 train-test ratio and normalizes the continuous features using `StandardScaler` to ensure mean=0 and variance=1.
5. **Artifact Generation:** Saves the processed data (`x_train.pkl`, etc.) and the scaler (`scaler.pkl`) for downstream consumption.

## 5. Model Training & Evaluation (`src/model_training.py`)
The project utilizes an ensemble learning approach for high accuracy and robustness against overfitting.
- **Algorithm:** `GradientBoostingClassifier` with tuned hyperparameters (`n_estimators=80`, `learning_rate=0.02`, `max_depth=5`, etc.).
- **Evaluation Metrics:** The model is rigorously evaluated on unseen test data. Metrics calculated include:
  - **Accuracy & Precision**
  - **Recall** (Crucial for healthcare to minimize false negatives)
  - **F1 Score** (Balance between precision and recall)
  - **ROC-AUC** (Model's ability to distinguish between classes)
- **Artifact Generation:** The trained model is saved as `model.pkl`.

## 6. Experiment Tracking with MLflow & DagsHub
To maintain strict version control over machine learning experiments, **MLflow** is integrated directly into the training script. 
- The tracking URI is configured to point to a remote **DagsHub** repository.
- Every time the training script runs, MLflow logs the hyperparameters used, the performance metrics achieved, and the specific git commit hash (if available).
- This ensures that any model can be audited and reproduced precisely.

## 7. Pipeline Orchestration with Kubeflow
An automated Directed Acyclic Graph (DAG) was constructed using the **Kubeflow Pipelines (KFP)** SDK (`kubeflow_pipeline/mlops_pipeline.py`).
- The pipeline defines two sequential containerized tasks: `data-processing` and `model-training`.
- The `model-training` task strictly depends on the successful execution of `data-processing` (`.after(data_processing_task)`).
- The python script compiles this logic into an `mlops_pipeline.yaml` file, which is executed on a Kubernetes cluster running the Kubeflow dashboard, ensuring isolated, scalable execution.

## 8. Deployment & Serving
To make the model accessible, a RESTful web service was developed.
- **Backend (`application.py`):** A Flask server that loads the `model.pkl` and `scaler.pkl`. It exposes a `/predict` endpoint that processes incoming form data, pads missing features (since the frontend requests 5 features while the scaler expects 10), and passes the vector to the model for inference.
- **Frontend (`templates/index.html` & `static/style.css`):** A responsive, premium web interface featuring a sleek minimalist monochrome design. It utilizes Jinja2 templating to dynamically render the model's prediction back to the user seamlessly.
- **Dockerization (`Dockerfile`):** The entire application is containerized using Docker. A `requirements.txt` file dictates dependencies, and the image is built and pushed to Docker Hub, allowing it to be spun up on any machine with a single command (`docker run -p 5000:5000 hhxcsa/mlopsapp:latest`).

## 9. Conclusion
The OncoPredict AI project successfully demonstrates the transition from a standard data science script to a fully realized MLOps product. By integrating Kubeflow for orchestration, MLflow for tracking, and Docker/Flask for deployment, the project guarantees reproducibility, scalability, and ease of access. This term project solidifies core competencies in machine learning, software engineering, and DevOps practices.
