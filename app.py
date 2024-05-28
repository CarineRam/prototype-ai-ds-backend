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
import uuid
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


DATASETS_DIR = 'datasets'
FEATURE_COLUMNS_DIR = 'feature_columns'

if not os.path.exists(FEATURE_COLUMNS_DIR):
    os.makedirs(FEATURE_COLUMNS_DIR)

def read_csv_file(file_path):
    return pd.read_csv(file_path)

@app.route('/datasets', methods=['GET'])
def list_datasets():
    datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    return jsonify(datasets)

@app.route('/get_dataset', methods=['POST'])
def get_dataset():
    dataset_name = request.form['dataset_name']
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        first_10_rows = df.head(10).to_json(orient='records')
        return jsonify({'data': first_10_rows})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

def load_feature_columns():
    if os.path.exists(FEATURE_COLUMNS_DIR):
        with open(FEATURE_COLUMNS_DIR, 'r') as f:
            return json.load(f)
    return []

def save_feature_columns(columns):
        with open(FEATURE_COLUMNS_DIR, 'w') as file:
            json.dump(columns, file)

@app.route('/save_columns', methods=['POST'])
def save_columns():
    data = request.json
    selected_columns = data.get('selected', [])
    
    print("Selected columns:", selected_columns)
    
    return jsonify({"success": True, "features": selected_columns})

@app.route('/get_features', methods=['GET'])
def get_features():
    feature_columns = load_feature_columns()
    return jsonify({"features": feature_columns})

if __name__ == '__main__':
    app.run( debug=True)