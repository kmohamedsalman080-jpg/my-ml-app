# 🚀 Server Overheat Prediction API — End-to-End MLOps Pipeline

An end-to-end Machine Learning API designed to predict server overheating risks, containerized using **Docker**, tested via **GitHub Actions CI/CD**, and live deployed on **Render Cloud Platform**.

---

## 🌟 Key Features

* **Machine Learning Model:** Scikit-Learn trained classification/regression model for server overheat prediction.
* **FastAPI Web Service:** Interactive RESTful API with Swagger UI documentation (`/docs`).
* **System Monitoring:** System resource status check endpoint (`/system-status`).
* **Docker Containerization:** Isolated, reproducible environment with small footprint.
* **Automated CI/CD Pipeline:** Automated build and unit tests on every GitHub push/PR using GitHub Actions.
* **Cloud Deployment:** Continuous deployment configured on Render.

---

## 🛠️ Tech Stack & Tools

| Component | Technology Used |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Framework** | FastAPI, Uvicorn |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud Hosting** | Render |

---

## 🌐 Live Demo & Endpoints

* **Live API Base URL:** `https://my-ml-app-r0y1.onrender.com`
* **Interactive API Docs (Swagger UI):** `https://my-ml-app-r0y1.onrender.com/docs`

### Key Endpoints:
1. `GET /` — Root health check and app status.
2. `GET /system-status` — Real-time CPU and Memory usage monitoring.
3. `POST /predict` — Server overheat risk prediction model inference.

---

## 💻 Local Setup & Installation

### Option 1: Running Locally with Python
```bash
# 1. Clone Repository
git clone [https://github.com/YOUR_GITHUB_USERNAME/my-ml-app.git](https://github.com/YOUR_GITHUB_USERNAME/my-ml-app.git)
cd my-ml-app

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Start Application
uvicorn main:app --reload