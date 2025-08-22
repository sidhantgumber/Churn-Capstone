# Churn Capstone Project

## Overview
This project is a comprehensive customer intelligence and churn prediction platform, leveraging machine learning, large language models (LLMs), and advanced analytics to provide actionable insights for business growth and customer retention. The solution is deployed on Google Cloud Platform (GCP) and accessible via:

**Service URL:** [https://churn-capstone-629741237893.us-central1.run.app](https://churn-capstone-629741237893.us-central1.run.app)

---

## Features
- **ML Model API**: Predicts customer churn, segments, and sales forecasts using multiple models (Logistic Regression, SVM, Decision Tree, Random Forest, Linear Regression, KMeans, Sentiment Analysis).
- **LLM Insights**: Generates deep customer intelligence insights using Mistral 7B and other LLMs.
- **Text-to-Speech (TTS)**: Converts insights into high-quality speech using Edge TTS.
- **Dashboards & Visualizations**: Power BI dashboard and CSV dashboards for churn, segments, feedback, and sales forecasting.
- **Data Analysis Notebooks**: Jupyter notebooks for EDA, clustering, churn prediction, text analysis, and sales forecasting.

---

## Project Structure
```
app.py                        # Main Flask API application
llm_insights.py               # LLM-powered customer insights generation
tts.py                        # Text-to-speech module for insights
customer_insights_mistral.txt # Example output from LLM (Mistral 7B)
requirements.txt              # Python dependencies
Dockerfile                    # Containerization for deployment
FinalProjDashboard.pbix       # Power BI dashboard
models/                       # Pre-trained ML models (pkl files)
data/                         # Datasets and dashboards (CSV files)
plots/                        # Cluster visualizations (PNG)
audio_output/                 # Generated audio files
notebooks/                    # Jupyter notebooks for analysis
```

---

## API Endpoints
- `/` : Health check and endpoint listing
- `/predict` : ML model predictions (churn, segments, sales)
- `/sentiment` : Sentiment analysis
- `/llm_insights` : Generate customer insights using LLM
- `/tts` : Convert insights to speech
- `/tts_insights` : Get insights audio directly

---

## Data & Models
- **Datasets**: Customer intelligence, clusters, feedback, sales forecasting, sentiment keywords, discovered topics, dashboards (churn, segments, feedback, sales).
- **Models**: Logistic Regression, SVM, Decision Tree, Random Forest, Linear Regression, KMeans, Sentiment (VADER), Scaler.
- **Visualizations**: Cluster plots (3, 4, 5 clusters), Power BI dashboard.

---

## Insights Example
See `customer_insights_mistral.txt` for a sample of LLM-generated insights, including:
- Sales performance trends
- Customer segment analysis
- Churn risk assessment
- Sales forecasting
- Actionable recommendations

---

## Deployment
- **Docker**: Containerized with Python 3.11, Flask, Gunicorn. See `Dockerfile` and `requirements.txt` for build/run instructions.
- **GCP Cloud Run**: Deployed using the following command:
  ```sh
  gcloud run deploy churn-capstone --image us-central1-docker.pkg.dev/churn-capstone/churn-capstone/churn-capstone --platform managed --region us-central1 --allow-unauthenticated --port 8080
  ```
- **Service URL**: [https://churn-capstone-629741237893.us-central1.run.app](https://churn-capstone-629741237893.us-central1.run.app)

---

## Getting Started
1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run locally**
   ```sh
   flask run --host=0.0.0.0 --port=8080
   ```
4. **Build Docker image**
   ```sh
   docker build -t churn-capstone .
   docker run -p 8080:8080 churn-capstone
   ```
5. **Deploy to GCP** (see above)

---

## Notebooks
- `notebooks/eda.ipynb` : Exploratory Data Analysis
- `notebooks/clustering.ipynb` : Customer segmentation
- `notebooks/churn_prediction.ipynb` : Churn modeling
- `notebooks/sales_forecasting.ipynb` : Sales prediction
- `notebooks/text_analysis.ipynb` : Text and sentiment analysis

---

## Power BI Dashboard
- `FinalProjDashboard.pbix` : Interactive dashboard for business users

---

## Audio Insights
- `audio_output/insights_from_file.mp3` : Example of insights converted to speech

---

## Authors & Credits
- Project by Sidhant Gumber
- ML, LLM, and TTS modules built using Python, scikit-learn, NLTK, Edge TTS, Flask
- Insights powered by Mistral 7B LLM

---

## License
This project is for educational and demonstration purposes.
