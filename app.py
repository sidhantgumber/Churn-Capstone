
from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np
import os
import logging


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
INSIGHTS_FILE = "customer_insights_mistral.txt"
AUDIO_FILE = "audio_output/insights_from_file.mp3"
with open('models/logistic_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('models/svm_rbf.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('models/decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/linreg_forecast.pkl', 'rb') as f:
    linreg_model = pickle.load(f)
with open('models/sentiment_vader.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)
with open('models/kmeans.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)


@app.route('/')
def home():
    return jsonify({
        "message": "ML Model API is running.",
        "endpoints": [
            "/predict - ML model predictions",
            "/sentiment - Sentiment analysis", 
            "/llm_insights - Generate LLM customer insights",
            "/tts - Convert insights to speech",
            "/tts_insights - Get insights audio directly"
        ]
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        features = data.get('features')
        if not model_type or features is None:
            return jsonify({"error": "Please provide 'model_type' and 'features' in the request body."}), 400
        data_np = np.array(features).reshape(1, -1)
        if model_type == "logreg":
            prediction = lr_model.predict(data_np)[0]
        elif model_type == "svm":
            prediction = svm_model.predict(data_np)[0]
        elif model_type == "dt":
            prediction = dt_model.predict(data_np)[0]
        elif model_type == "rf":
            prediction = rf_model.predict(data_np)[0]
        elif model_type == "linreg":
            prediction = linreg_model.predict(data_np)[0]
        elif model_type == "kmeans":
            prediction = kmeans_model.predict(data_np)[0]
        else:
            return jsonify({"error": "Invalid model type. Choose 'logreg', 'svm', 'dt', 'rf', 'linreg', or 'kmeans'."}), 400
        return jsonify({
            "model_type": model_type,
            "features": features,
            "prediction": float(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sentiment', methods=['POST'])
def sentiment():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({"error": "Please provide 'text' in the request body."}), 400
        scores = sentiment_model.polarity_scores(text)
        return jsonify({
            "text": text,
            "scores": scores
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# @app.route('/llm_insights', methods=['POST'])
# def llm_insights():
#     try:
#         data = request.get_json() if request.is_json else {}
#         regenerate = data.get('regenerate', False)
#         if os.path.exists(INSIGHTS_FILE) and not regenerate:
#             logger.info("Using existing insights file")
#             with open(INSIGHTS_FILE, 'r', encoding='utf-8') as f:
#                 insights = f.read()
#         else:
#             logger.info("Running llm_insights.py to generate new insights...")
#             result = subprocess.run(
#                 ['python', 'llm_insights.py'], 
#                 capture_output=True, 
#                 text=True,
#                 timeout=600
#             )
#             if result.returncode != 0:
#                 logger.error(f"llm_insights.py failed: {result.stderr}")
#                 return jsonify({
#                     "error": f"Failed to generate insights: {result.stderr}"
#                 }), 500
#             if os.path.exists(INSIGHTS_FILE):
#                 with open(INSIGHTS_FILE, 'r', encoding='utf-8') as f:
#                     insights = f.read()
#                 logger.info("Insights generated successfully")
#             else:
#                 return jsonify({
#                     "error": "Insights file not generated"
#                 }), 500
        
#         return jsonify({
#             "status": "success",
#             "insights": insights,
#             "insights_file": INSIGHTS_FILE,
#             "message": "Insights generated successfully"
#         })
#     except subprocess.TimeoutExpired:
#         return jsonify({"error": "Insights generation timed out"}), 500
#     except Exception as e:
#         logger.error(f"Error generating insights: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/tts_insights', methods=['GET', 'POST'])
# def tts_insights():
#     try:
#         data = request.get_json() if request.is_json else {}
#         voice = data.get('voice', 'en-US-AriaNeural')
#         regenerate_insights = data.get('regenerate_insights', False)
#         regenerate_audio = data.get('regenerate_audio', False)
#         if os.path.exists(AUDIO_FILE) and not regenerate_audio:
#             logger.info("Using existing audio file")
#             return send_file(
#                 AUDIO_FILE,
#                 mimetype='audio/mpeg',
#                 as_attachment=True,
#                 download_name='customer_insights.mp3'
#             )

#         if not os.path.exists(INSIGHTS_FILE) or regenerate_insights:
#             logger.info("Generating insights first...")
#             result = subprocess.run(
#                 ['python', 'llm_insights.py'], 
#                 capture_output=True, 
#                 text=True,
#                 timeout=600
#             )
#             if result.returncode != 0:
#                 return jsonify({
#                     "error": f"Failed to generate insights: {result.stderr}"
#                 }), 500

#         if not os.path.exists(INSIGHTS_FILE):
#             return jsonify({
#                 "error": f"Insights file not found: {INSIGHTS_FILE}"
#             }), 404
#         audio_path = generate_audio_from_file(
#             INSIGHTS_FILE, 
#             voice=voice, 
#             output_file=AUDIO_FILE
#         )
#         logger.info(f"Audio generated: {audio_path}")
#         return send_file(
#             audio_path,
#             mimetype='audio/mpeg',
#             as_attachment=True,
#             download_name='customer_insights.mp3'
#         )
#     except subprocess.TimeoutExpired:
#         return jsonify({"error": "Insights generation timed out"}), 500
#     except Exception as e:
#         logger.error(f"TTS Insights Error: {str(e)}")
#         return jsonify({"error": str(e)}), 500

@app.route('/llm_insights', methods=['POST'])
def llm_insights():
    try:
        data = request.get_json() if request.is_json else {}
        regenerate = data.get('regenerate', False)
        
        if os.path.exists(INSIGHTS_FILE) and not regenerate:
            logger.info("Using existing insights file")
            with open(INSIGHTS_FILE, 'r', encoding='utf-8') as f:
                insights = f.read()
        else:
            return jsonify({
                "error": "Insights file not found. Generation disabled in production."
            }), 404
        
        return jsonify({
            "status": "success",
            "insights": insights,
            "insights_file": INSIGHTS_FILE,
            "message": "Pre-generated insights served successfully"
        })
        
    except Exception as e:
        logger.error(f"Error serving insights: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/tts_insights', methods=['GET', 'POST'])
def tts_insights():
    try:
        data = request.get_json() if request.is_json else {}
        # voice = data.get('voice', 'en-US-AriaNeural')  # Not needed anymore
        # regenerate_insights = data.get('regenerate_insights', False)  # Not needed
        # regenerate_audio = data.get('regenerate_audio', False)  # Not needed
        
        if os.path.exists(AUDIO_FILE):
            logger.info("Using existing audio file")
            return send_file(
                AUDIO_FILE,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name='customer_insights.mp3'
            )
        else:
            return jsonify({
                "error": f"Audio file not found: {AUDIO_FILE}. TTS generation disabled in production."
            }), 404
        
    except Exception as e:
        logger.error(f"TTS Insights Error: {str(e)}")
        return jsonify({"error": str(e)}), 500




# if __name__ == '__main__':
#     print("Starting ML Model API...")
#     if os.path.exists(INSIGHTS_FILE):
#         print(f"Insights file found: {INSIGHTS_FILE}")
#     else:
#         print(f"Insights file missing: {INSIGHTS_FILE}")
#     if os.path.exists(AUDIO_FILE):
#         print(f"Audio file found: {AUDIO_FILE}")
#     else:
#         print(f"Audio file missing: {AUDIO_FILE}")
#     app.run(host='0.0.0.0', port=5000, debug=False)
"""
SAMPLE REQUEST BODIES FOR POSTMAN TESTING 

For /predict endpoint (replace feature values with real data as needed):

Logistic Regression, SVM, Decision Tree, Random Forest:
{
  "model_type": "logreg",  # or "svm", "dt", "rf"
  "features": [2, 41, 185, 249000.0, 124500.0, 225000.0, 3.5, 5.0, 28500.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 2, 2, 0.5, 1, 0, 1, 0.0, 0, 2, 1.0, 43, 0.5, 124500.0, 1.4634146341463414, 182195.1219512195, 0, 0, 1, 0, 0, 0, 0, 1]
}

KMeans:
{
  "model_type": "kmeans",
  "features": [5.23, -0.1]  
}
Linear Regression (Sales Forecasting):
{
  "model_type": "linreg",
  "features": [8, 3, 0.5, 0.866, 12, 2, 1, 100, 25000, 3, 100, 0.05, 0.02, 200, 50, 1000, 45, 5000, 0.8, 0.1, 0.1]
  // Replace with actual monthly feature values in correct order
}

For /sentiment endpoint:
{
  "text": "I love this product! It works perfectly and the support is great."
}
"""