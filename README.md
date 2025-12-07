🚀 Machine Failure Prediction API

A production-ready FastAPI-based Machine Learning API that predicts whether an industrial machine is likely to fail, using sensor data and logistic regression.

Live API: https://machine-failure-api.onrender.com

Interactive Docs (Swagger UI): https://machine-failure-api.onrender.com/docs

📌 Overview

This project turns a machine learning model into a fully deployed, cloud-hosted API using:

FastAPI (backend)

scikit-learn (ML model)

Render (cloud deployment)

Joblib (model persistence)

JSON-based prediction endpoints

The API receives machine parameters and returns:

Prediction (0 = No Failure, 1 = Failure)

Failure probability

Normal operating probability

🧠 Machine Learning Model

The dataset includes:

Rotational speed

Torque

Tool wear

Thermal & mechanical failure flags (TWF, HDF, PWF, OSF)

The model is trained using Logistic Regression, standardized via StandardScaler, and saved as:

model.pkl
scaler.pkl

🛠 Tech Stack
Component	Technology
Backend	FastAPI
Model	Scikit-learn Logistic Regression
Deployment	Render
Format	REST API (JSON)
📡 API Endpoints
1️⃣ Root

GET /
Returns API status.

2️⃣ Predict Machine Failure

POST /predict

Request Body
{
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 10,
  "twf": 0,
  "hdf": 0,
  "pwf": 0,
  "osf": 0
}

Response
{
  "prediction": 0,
  "probability_alert": 0.0491,
  "probability_normal": 0.9508
}

📁 Project Structure
machine_failure_api/
│── main.py
│── train_model.py
│── model.pkl
│── scaler.pkl
│── machine_failure.csv
│── requirements.txt
│── render.yaml

🚀 Deployment (Render)

Render automatically:

✔ Installs dependencies
✔ Loads model artifacts
✔ Runs FastAPI using Uvicorn

Configured using:

render.yaml

👩‍💻 Author

Riya Shende
MSc Artificial Intelligence & Machine Learning Engineer
📍 London, UK
🔗 LinkedIn: https://www.linkedin.com/in/iriya-shende/

🎉 This API is live and production-ready!
