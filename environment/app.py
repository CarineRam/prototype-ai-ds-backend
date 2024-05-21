from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
from io import StringIO
import matplotlib.pyplot as plt
from model import load_model
import io
import os

app = Flask(__name__)
CORS(app)
# model = load_model()
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         text = data['text']

#         predicted_class = model.predict(text)
#         confidence = max(model.predict_proba(text))

#         return jsonify({"predicted_class": int(predicted_class), "confidence": float(confidence)})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
 
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         text = data['text']
#         predicted_class = model.predict([text])[0]
#         confidence = max(model.predict_proba([text])[0])
#         return jsonify({"predicted_class": int(predicted_class), "confidence": float(confidence)})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file provided"}), 400

        # Assume the uploaded file is a CSV
        data = pd.read_csv(StringIO(file.read().decode('utf-8')))

        if 'text' not in data.columns or 'label' not in data.columns:
            return jsonify({"error": "Invalid file format. Ensure the file has 'text' and 'label' columns."}), 400

        X, y = data['text'], data['label']
        model = make_pipeline(TfidfVectorizer(), LogisticRegression())
        model.fit(X, y)

        joblib.dump(model, 'text_classification_model.joblib')

        return jsonify({"message": "Model trained successfully with the uploaded data"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# @app.route('/class_distribution', methods=['GET'])
# def class_distribution():
#     data = {
#         'classes': ['Class 1', 'Class 2', 'Class 3'],
#         'counts': [12, 19, 3]
#     }
#     return jsonify(data)


@app.route('/histogram', methods=['GET'])
def histogram():
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    plt.hist(data, bins = 4)
    plt.xlabel('Valeurs')
    plt.ylabel('Fq')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')


# @app.route('/precision_recall', methods=['GET'])
# def precision_recall():
#     data = {
#         'precision': [0.1, 0.5, 0.7],
#         'recall': [0.3, 0.6, 0.8],
#         'labels': ['Point 1', 'Point 2', 'Point 3']
#     }
#     return jsonify(data)

# @app.route('/confusion_matrix', methods=['GET'])
# def confusion_matrix():
#     data = {
#         'matrix': [
#             [65, 10, 5],
#             [10, 75, 15],
#             [5, 15, 85]
#         ],
#         'labels': ['Class 1', 'Class 2', 'Class 3']
#     }
#     return jsonify(data)

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({"status": "OK"})

if __name__ == '__main__':
    app.run( debug=True)