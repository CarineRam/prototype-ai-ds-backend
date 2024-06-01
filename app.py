from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
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

DATASETS_DIR = 'datasets'
FEATURE_COLUMNS_FILE = 'feature_columns.json'
SELECTED_MODEL_FILE = 'selected_model.json'
dataset_name = None

models = {
        "LogisticRegression" : {
            "type": "Classification", 
            "hyperparameters": {
                "C": 1.0, 
                "solver": "lbfgs"
            }
        },
        "SVC": {
            "type": "Classification",
            "description": "SVC is a supervised learning algorithm used for classification. It finds an optimal hyperplane in an N-dimensional space (where N is the number of features) that separates the classes.",
            "hyperparameters": {
                "C": "float (default=1.0)",
                "kernel": "str (default='rbf')",
                "gamma": "float (default='scale')",
            }
        },
        "MLPClassifier": {
            "type": "Classification",
            "description": "MLPClassifier is a type of artificial neural network known as a multi-layer perceptron. It can be used for classification and regression tasks.",
            "hyperparameters": {
                "hidden_layer_sizes": "tuple (default=(100,))",
                "activation": "str (default='relu')",
                "solver": "str (default='adam')",
                "alpha": "float (default=0.0001)",
            }
        },
        "GaussianNB": {
            "type": "Classification",
            "description": "GaussianNB is a simple classification algorithm based on Bayes' theorem with a Gaussian naive assumption. It assumes that features are independent and follow a normal distribution.",
            "hyperparameters": {}
        },
        "MultinomialNB": {
            "type": "Classification",
            "description": "MultinomialNB is a variant of the Naive Bayes algorithm suitable for data with multinomial distributions, such as word counts in text data.",
            "hyperparameters": {
                "alpha": "float (default=1.0)",
                "fit_prior": "bool (default=True)",
            }
        }
    }

if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

# if not os.path.exists(FEATURE_COLUMNS_FILE):
#     os.makedirs(FEATURE_COLUMNS_FILE)

# if not os.path.exists(FEATURE_COLUMNS_FILE):
#     with open(FEATURE_COLUMNS_FILE, 'w') as f:
#         json.dump([], f)

for file in (FEATURE_COLUMNS_FILE, SELECTED_MODEL_FILE):
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump({}, f)

def read_csv_file(file_path):
    return pd.read_csv(file_path)

@app.route('/datasets', methods=['GET'])
def list_datasets():
    datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    return jsonify(datasets)

@app.route('/get_dataset', methods=['POST'])
def get_dataset():
    global dataset_name
    dataset_name = request.form['dataset_name']
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        first_10_rows = df.head(10).to_json(orient='records')
        return jsonify({'data': first_10_rows})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

def load_feature_columns():
    if os.path.exists(FEATURE_COLUMNS_FILE):
        with open(FEATURE_COLUMNS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feature_columns(columns):
        with open(FEATURE_COLUMNS_FILE, 'w') as file:
            json.dump(columns, file)

@app.route('/save_columns', methods=['POST'])
def save_columns():
    data = request.json
    selected_columns = data.get('selected', [])
    
    print("Selected columns:", selected_columns)

    save_feature_columns(selected_columns)
    
    return jsonify({"success": True, "features": selected_columns})

@app.route('/get_features', methods=['GET'])
def get_features():
    feature_columns = load_feature_columns()
    return jsonify({"features": feature_columns})

@app.route('/get_unselected_columns', methods=['POST'])
def get_unselected_columns():
    dataset_name = request.form['dataset_name']
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        all_columns = set(df.columns)
        selected_columns = set(load_feature_columns())
        unselected_columns = list(all_columns - selected_columns)
        print("Unselected columns:", unselected_columns)
        return jsonify({'unselected_columns': unselected_columns})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

 
#Model choices
def save_selected_model(model_name):
    with open(SELECTED_MODEL_FILE, 'w') as f:
        json.dump({'model': model_name}, f)

def load_selected_model():
    if os.path.exists(SELECTED_MODEL_FILE):
        with open(SELECTED_MODEL_FILE, 'r') as f:
            return json.load(f).get('model', '')
    return ''

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': list(models.keys())})

@app.route('/select_model', methods=['POST'])
def select_model():
    data = request.json
    selected_model = data.get('model')
    save_selected_model(selected_model)
    print("The selected model:", selected_model)
    return jsonify({'success': True, 'selected_model': selected_model})

@app.route('/model_details', methods=['POST'])
def get_model_details():
    model_name = request.json.get('model')
    if model_name:
        if model_name in models:
            return jsonify(models[model_name])
        else:
            return jsonify({"error": "Model not found"}), 404
    else:
        return jsonify({"error": "Model name not provided"}), 400
    
#Post the percentage of split data
@app.route('/split_data',methods=['POST'])
def split_data():
    global dataset_name

    try:
        if dataset_name is None:
            return jsonify({'error': 'Dataset name not provided'}), 400

        dataset_path = os.path.join(DATASETS_DIR, dataset_name)
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            return jsonify({'error': 'Dataset not found'}), 400

        data = request.json
        train_percentage = data.get('trainPercentage', 0)
        test_percentage = data.get('testPercentage', 0)

        train, test = train_test_split(df, test_size=test_percentage, random_state=4)

        response_data = {
            'message': 'Data split successfully',
            'trainPercentage': train_percentage,
            'testPercentage': test_percentage
        }

        return jsonify({'train_size': len(train), 'test_size': len(test)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run( debug=True)