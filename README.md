# 🛠️ Automated MLOps Pipeline for Predictive Maintenance

[![MLOps Pipeline](https://github.com/hammad070707/Predictive_Maintenance_MLOps/actions/workflows/main.yml/badge.svg)](https://github.com/hammad070707/Predictive_Maintenance_MLOps/actions)

## 📌 Project Overview
This project demonstrates a production-grade, end-to-end Machine Learning pipeline for **Predictive Maintenance**. Using the UCI AI4I 2020 Predictive Maintenance Dataset, the system predicts machine failures (No Failure vs. Failure) based on sensor data like temperature, RPM, and torque.

The core focus of this project is **MLOps (Machine Learning Operations)**—ensuring that the model is not just a static script but a living system that automates data versioning, experiment tracking, model registration, and containerized deployment.

---

## 🏗️ System Architecture
The pipeline follows a closed-loop MLOps workflow:
1. **Data Ingestion:** Automated loading of raw sensor data.
2. **Preprocessing:** Handling imbalanced classes, label encoding, and feature scaling.
3. **Experiment Tracking (MLflow):** Logging hyperparameters (Random Forest) and metrics (Accuracy, F1-Score).
4. **Model Registry:** Automatic comparison of "Champion" vs "Challenger" models. Only superior models are registered for production.
5. **API Serving:** Model deployment via **FastAPI** with a Pydantic schema for data validation.
6. **Containerization:** The entire environment is packaged using **Docker** for portability.
7. **CI/CD:** Automated testing and Docker building via **GitHub Actions** on every push.

---

## 🚀 Tech Stack
*   **Language:** Python 3.11
*   **Machine Learning:** Scikit-Learn (Random Forest Classifier)
*   **Tracking & Registry:** MLflow
*   **Data Versioning:** DVC (Data Version Control)
*   **API Framework:** FastAPI & Uvicorn
*   **Containerization:** Docker
*   **Automation:** GitHub Actions (CI/CD)

---

## 📊 Model Performance
To handle class imbalance (failures are rare), the model was tuned using `class_weight='balanced'`. 
- **Target Accuracy:** ~94% - 98%
- **Target F1-Score:** ~0.70+ 
- **Tracking:** All experiments are logged in the MLflow UI for reproducibility.

---

## 🛠️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/hammad070707/Predictive_Maintenance_MLOps.git
cd Predictive_Maintenance_MLOps

2. Set up Virtual Environment

python -m venv venv
source venv/Scripts/activate  # On Windows
pip install -r requirements.txt

3. Run the Pipeline

# Train and Evaluate the model
python src/model_training.py
python src/model_evaluation.py

4. Start the API (Local)

python app.py
Visit http://127.0.0.1:8000/docs to test the API via Swagger UI.

🐳 Docker Deployment
To run the application in a completely isolated environment:

# Build the Docker image
docker build -t predictive-maintenance-app .

# Run the container

docker run -p 8000:8000 predictive-maintenance-app

🤖 CI/CD Workflow

Every time code is pushed to the main branch, GitHub Actions:
Sets up a fresh Ubuntu environment.
Installs all dependencies.
Runs the entire ML Pipeline (Ingestion -> Preprocessing -> Training -> Evaluation).
Validates the build.
Builds a new Docker Image.
## 👨‍💻 Author
**Hammad Ahmed**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hammad-ahmed-ai/)  
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hammad070707)

---
*This project was built to demonstrate that Machine Learning is not just about notebooks, but about building robust, automated systems.*
