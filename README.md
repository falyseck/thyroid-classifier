# 🔬 Thyroid Cancer Image Classifier

Binary classification of thyroid ultrasound images — **Benign vs Malignant** — using EfficientNetB0 transfer learning, deployed as a FastAPI + Streamlit pipeline on Render.

---

## 🎥 Demo

- **Video Demo:** [YouTube Link — add after recording]
- **Live API:** [https://thyroid-classifier-api1.onrender.com](https://thyroid-classifier-api.onrender.com)
- **Live UI:** [https://thyroid-classifier-ui.streamlit.app/](https://thyroid-classifier-ui.streamlit.app/)
- **API Docs:** [https://thyroid-classifier-api1.onrender.com/docs](https://thyroid-classifier-api1.onrender.com/docs)

---

## 📋 Project Description

This project implements an end-to-end Machine Learning pipeline for thyroid cancer classification from ultrasound images. It covers:

- **Data Acquisition** — Thyroid ultrasound dataset from Kaggle (~2879 images, 2 classes)
- **Preprocessing** — EfficientNet-native preprocessing, augmentation, 70/15/15 split
- **Model Development** — EfficientNetB0 with transfer learning; v1 (broken) → v2 (overfitting) → v3 (regularised, production-ready)
- **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix, optimal threshold tuning
- **API** — FastAPI with predict, upload, retrain, and health endpoints
- **UI** — Streamlit dashboard with dark medical aesthetic
- **Deployment** — Dockerised, deployed on Render
- **Load Testing** — Locust flood simulation

### Model Performance (v3 — Final)

| Metric | Score |
|---|---|
| AUC-ROC | ~0.80 |
| Accuracy | ~68% (val) |
| Recall | ~72% |
| F1 Score | ~68% |

> Val accuracy is intentionally modest — v3 prioritises generalisation over memorisation. Train/val gap was reduced from 18% (v2) to ~8% (v3) through regularisation.

---

## 🗂 Directory Structure

```
thyroid_classifier/
│
├── README.md
├── app.py                    ← Streamlit UI
├── Dockerfile.api            ← Docker image for FastAPI
├── Dockerfile.ui             ← Docker image for Streamlit
├── render.yaml               ← Render deployment config
│
├── notebook/
│   └── thyroid_cancer_classification.ipynb
│
├── src/
│   ├── preprocessing.py      ← Data loading, splitting, augmentation
│   ├── model.py              ← Model build, train, retrain
│   └── prediction.py         ← Inference functions
│
├── api/
│   ├── main.py               ← FastAPI app
│   └── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── benign/
│   │   └── malignant/
│   ├── train/
│   ├── val/
│   └── test/
│
└── models/
    ├── thyroid_efficientnet.h5
    ├── thyroid_efficientnet.tf
    └── model_meta.pkl
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- Docker (for containerised deployment)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/thyroid-classifier.git
cd thyroid-classifier
```

### 2. Install dependencies
```bash
# API dependencies
pip install -r api/requirements.txt

# UI dependencies
pip install streamlit plotly requests Pillow numpy
```

### 3. Prepare the dataset
Download from Kaggle and place images in:
```
data/raw/benign/      ← benign .jpg images
data/raw/malignant/   ← malignant .jpg images
```

### 4. Run the notebook
Open and run `notebook/thyroid_cancer_classification.ipynb` end-to-end.
This will:
- Split the dataset (70/15/15)
- Train EfficientNetB0 (v2 + v3)
- Save `models/thyroid_efficientnet.h5` and `models/model_meta.pkl`

### 5. Start the API
```bash
# From project root
uvicorn api.main:app --reload --port 8000
```
Test at: http://localhost:8000/docs

### 6. Start the Streamlit UI
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

Set API URL to `http://localhost:8000` and click **Connect**.

---

## 🐳 Docker (Local)

```bash
# Build and run API
docker build -f Dockerfile.api -t thyroid-api .
docker run -p 8000:8000 thyroid-api

# Build and run UI
docker build -f Dockerfile.ui -t thyroid-ui .
docker run -p 8501:8501 thyroid-ui
```

---

## ☁️ Deploy on Render

### Step-by-step

1. **Push to GitHub**
```bash
git add .
git commit -m "initial commit"
git push origin main
```

2. **Deploy the API on Render**
   - Go to [render.com](https://render.com) → New → Web Service
   - Connect your GitHub repo
   - Select **Docker** as runtime
   - Set Dockerfile path: `./Dockerfile.api`
   - Set name: `thyroid-classifier-api`
   - Click **Deploy**
   - Copy the URL once deployed (e.g. `https://thyroid-classifier-api.onrender.com`)

3. **Deploy the UI on Render**
   - Go to Render → New → Web Service
   - Connect same GitHub repo
   - Set Dockerfile path: `./Dockerfile.ui`
   - Set name: `thyroid-classifier-ui`
   - Add environment variable: `API_URL = https://thyroid-classifier-api.onrender.com`
   - Click **Deploy**

4. **Verify**
   - API health: `https://thyroid-classifier-api.onrender.com/health`
   - API docs: `https://thyroid-classifier-api.onrender.com/docs`
   - UI: `https://thyroid-classifier-ui.onrender.com`

> ⚠️ Free tier Render services spin down after 15 minutes of inactivity. First request after sleep may take ~30 seconds.

---

## 🌊 Load Testing with Locust

### Install
```bash
pip install locust
```

### Run
```bash
locust -f locustfile.py --host=https://thyroid-classifier-api.onrender.com
```
Open http://localhost:8089, set number of users and spawn rate, then start.

### Results
See `locust_results/` folder for screenshots and CSV reports showing latency and response time with different Docker container counts.

---

## 🔁 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health ping |
| GET | `/health` | Detailed system status |
| GET | `/model-info` | Model version, metrics, threshold |
| GET | `/metrics` | Evaluation metrics |
| POST | `/predict` | Predict single image |
| POST | `/upload-data` | Upload ZIP of new images |
| POST | `/retrain` | Trigger retraining |
| GET | `/retrain-status` | Check retraining progress |
| GET | `/stats` | API usage statistics |

---

## 📊 Load Test Results

"users","requests","failures","median_ms","p95_ms","p99_ms","avg_ms","rps"
"5","25","0","0","49000","54000","0","0.318038475765688"
"10","68","0","0","11000","14000","0","0.783974306840002"
"20","65","0","0","16000","19000","0","0.748361821683752"
"40","59","0","0","44000","46000","0","0.662660250348718"


---

## 👤 Author

**El Hadji Faly Seck**  
African Leadership University — BSE  
Machine Learning Pipeline — Summative Assignment
