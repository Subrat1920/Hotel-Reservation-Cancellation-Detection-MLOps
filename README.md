# Hotel Reservation Prediction

**Project Goal:**  
Predict which customers are likely to **cancel their reservation or checkout before check-in** using historical hotel booking data. This enables hotels to take proactive actions to reduce revenue loss.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture & Pipeline](#architecture--pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contribution](#contributing)


---

## Project Overview
The Hotel Reservation Prediction project is an end-to-end machine learning solution for predicting early checkouts or cancellations. The pipeline automates data ingestion, preprocessing, model training, and deployment using modern CI/CD practices. A Flask-based web app allows interactive prediction requests via Docker deployment.

---

## Features
- Predict customer checkout behavior before check-in.
- ML model trained with **LightGBM**.
- Automated preprocessing and feature engineering.
- Experiment tracking with **MLflow**.
- Fully automated CI/CD pipeline using **Jenkins**, **Docker**, and **Google Cloud Platform (GCP)**.
- Integration with **GCP Buckets** and **Container Registry**.
- Role-based access with **IAMs and service accounts**.
- Flask web app for serving predictions in real-time.

---

## Tech Stack
- **Programming:** Python  
- **Data Processing & ML:** Pandas, NumPy, LightGBM, Scikit-learn  
- **Web App:** Flask  
- **CI/CD:** Jenkins, Docker, GitHub  
- **Cloud:** Google Cloud Platform (GCP) – Storage Buckets, Container Registry  
- **Experiment Tracking:** MLflow  
- **Version Control:** Git & GitHub  

---

## Architecture & Pipeline
1. **Data Ingestion:** Raw hotel booking data stored in GCP buckets is ingested via Python scripts.  
2. **Data Processing:** Cleaning, preprocessing, and feature engineering applied to prepare training datasets.  
3. **Model Training:** LightGBM model is trained, evaluated, and saved to `artifacts/models`.  
4. **MLflow Tracking:** All experiments, metrics, and models are tracked for reproducibility.  
5. **Deployment:**  
   - Flask app serves predictions.  
   - Docker image built and pushed to **GCP Container Registry**.  
   - Deployed with full CI/CD pipeline using Jenkins.  
6. **Automation & Security:**  
   - Jenkins pipelines handle automated builds, tests, and deployments.  
   - IAM roles and service accounts secure GCP resources.

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Subrat1920/Hotel-Reservation-Cancellation-Detection-MLOps.git
cd Hotel-Reservation-Prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables for GCP authentication (if using service accounts):
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

## Usage
### A. Local Execution (Training & Prediction)

1. Start the Flask app:
```bash
python app.py
```

2. Access the app at http://localhost:5000 to make predictions.

### B. Dockerized Deployment

1. Build the Docker image:
```bash
docker build -t hotel-reservation-prediction .
```

2. Run the container:
```bash
docker run -p 8080:8080 hotel-reservation-prediction
```

3. Access at http://localhost:8080.

5. CI/CD Pipeline
Jenkinsfile handles automated cloning, testing, building, pushing Docker image to GCP, and deploying.

## Project Structure
```bash
├── 📁 artifacts/
│   ├── 📁 models/
│   │   └── 📄 lgbm_model.pkl
│   ├── 📁 processed/
│   │   ├── 📄 process_test.csv
│   │   └── 📄 process_train.csv
│   └── 📁 raw/
│       ├── 📄 raw.csv
│       ├── 📄 test.csv
│       └── 📄 train.csv
├── 📁 config/
│   ├── 🐍 __init__.py
│   ├── ⚙️ config.yaml
│   ├── 🐍 model_params.py
│   └── 🐍 path_config.py
├── 📁 custom_jenkins/
│   └── 🐳 Dockerfile
├── 📁 pipeline/
│   ├── 🐍 __init__.py
│   └── 🐍 training_pipeline.py
├── 📁 src/
│   ├── 🐍 __init__.py
│   ├── 🐍 data_ingestion.py
│   ├── 🐍 data_processing.py
│   ├── 🐍 exception.py
│   ├── 🐍 logger.py
│   └── 🐍 model_training.py
├── 📁 static/
│   └── 🎨 style.css
├── 📁 templates/
│   └── 🌐 index.html
├── 📁 utils/
│   ├── 🐍 __init__.py
│   └── 🐍 common_functions.py
├── 🚫 .gitignore
├── 🐳 Dockerfile
├── 📄 Jenkinsfile
├── 📖 README.md
├── 🐍 app.py
├── 📄 requirements.txt
└── 🐍 setup.py
```

## Contributing

```bash
Fork the repository.
Create a feature branch: git checkout -b feature-name
Commit your changes: git commit -m "Add feature"
Push to branch: git push origin feature-name
Create a Pull Request.
```
---