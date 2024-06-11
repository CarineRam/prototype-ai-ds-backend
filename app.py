from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from io import StringIO
import matplotlib.pyplot as plt
from model import load_model
from io import BytesIO
import os
import uuid
import json
import matplotlib
matplotlib.use('Agg')
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

DATASETS_DIR = 'datasets'
FEATURE_COLUMNS_FILE = 'feature_columns.json'
SELECTED_MODEL_FILE = 'selected_model.json'
HISTOGRAMS_DIR = 'histograms'
HEATMAP_DIR ='heatmap'
PRECISION_RECALL_DIR = 'precision_recall'

dataset_name = None
train_percentage = None
test_percentage = None
selected_model = None
model = None
X_test = None
y_test = None
results = None
predicted = None

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

if not os.path.exists(HISTOGRAMS_DIR):
    os.makedirs(HISTOGRAMS_DIR)

if not os.path.exists(HEATMAP_DIR):
    os.makedirs(HEATMAP_DIR)

if not os.path.exists(PRECISION_RECALL_DIR):
    os.makedirs(PRECISION_RECALL_DIR)

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
    global unselected_columns
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
    print("Received data:", data)
    selected_model = data.get('model')

    if not selected_model:
        return jsonify({'error':'No model provided'}), 400
    
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

#train the model
@app.route('/train_model', methods=['POST'])
def train_model():
        global dataset_name, unselected_columns, train_percentage, test_percentage, X_test, y_test, model, results, predicted

    # try:
        if dataset_name is None:
            return jsonify({'error':'Dataset name not provide'}), 400
        
        dataset_path = os.path.join(DATASETS_DIR, dataset_name)
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            return jsonify({'error': 'Dataset not found'}), 400
        
        feature_columns = load_feature_columns()
        if not feature_columns:
            return jsonify({'error': 'No feature columns selected'}), 400
        
        if not unselected_columns:
            return jsonify({'error': 'Unselected columns not found'}), 400
        
        target_column = unselected_columns[0]
        if target_column not in df.columns:
            return jsonify({'error': 'Target column not found in dataset'}), 400
        
        X = df[feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=4)

        model_name = load_selected_model()
        if not model_name:
            return jsonify({'error': 'No model selected'}), 400
        
        model = None
        if model_name == 'LogisticRegression':
            model = LogisticRegression(**models[model_name]['hyperparameters'])
        elif model_name == 'SVC':
            model = SVC(**models[model_name]['hyperparameters'])
        elif model_name == 'MLPClassifier':
            model = MLPClassifier(**models[model_name]['hyperparameters'])
        elif model_name == 'GaussianNB':
            model = GaussianNB()
        elif model_name == 'MultinomialNB':
            model = MultinomialNB(**models[model_name]['hyperparameters'])
        else:
            return jsonify({'error': 'Selected model is not supported'}), 400

        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)
        print("after get dummies", X_train, X_test)
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        results = confusion_matrix(y_test, predicted) 

        print ('Confusion Matrix :')
        print(results) 
        score = accuracy_score(y_test, predicted)
        print ('Accuracy Score :', score)
        print ('Report : ')
        print( classification_report(y_test, predicted) )

        model_filename = f"trained_model_{uuid.uuid4().hex}.pkl"
        joblib.dump(model, os.path.join(DATASETS_DIR, model_filename))

        response_data = {
            'message': 'Model trained successfully',
            'model': model_name,
            'accuracy': score,
            'model_filename': model_filename,
        }

        return jsonify(response_data), 200

#generate an histogram
@app.route('/generate_histogram', methods=['POST'])
def generate_histogram():
        global model, X_test, y_test, results

        print(X_test)
        print(y_test)

    # try:
        if model is None or X_test is None or y_test is None:
            return jsonify({'error': 'Model or test data not found'}), 400

        plt.figure()
        plt.hist(y_test, label = 'Actual')
        plt.hist(model.predict(X_test), bins = 10, alpha = 0.5, label = 'Predicted')
        plt.legend(loc='upper right')
        plt.xlabel('Classes')
        plt.ylabel('Frequency')
        plt.title('Histogram of actual vs Predicted Classes')
        histogram_path = os.path.join(HISTOGRAMS_DIR, f"histogram_{uuid.uuid4().hex}.png")
        plt.savefig(histogram_path)

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')

#generate a heatmap
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
        global model, X_test, y_test, results

        if model is None or X_test is None or y_test is None:
            return jsonify({'error': 'Model or test data not found'}), 400

        plt.figure()
        sns.heatmap(results)
        plt.title('Heatmap')
        heatmap_path = os.path.join(HEATMAP_DIR, f"heatmap_{uuid.uuid4().hex}.png")
        plt.savefig(heatmap_path)

        img_hm = BytesIO()
        plt.savefig(img_hm, format='png')
        img_hm.seek(0)
        plt.close()

        return send_file(img_hm, mimetype='image/png')

#generate a precision recall curve
@app.route('/generate_precision_recall', methods=['POST'])
def generate_precision_recall():
    # if request.method == 'POST':
        global model, X_test, y_test, results, predicted

        # if model is None or model not in models:
        #     return jsonify({'error': 'Invalid or no model selected'}), 400

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            return jsonify({'error': 'Model does not support predict_proba or decision_function'}), 400
        
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        average_precision = average_precision_score(y_test, y_scores)

        plt.figure()
        plt.plot(recall, precision, marker='.', label=f'{model.__class__.__name__} (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        precision_recall_path = os.path.join(PRECISION_RECALL_DIR, f"precision_recall_{uuid.uuid4().hex}.png")
        plt.savefig(precision_recall_path, format='png')

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')
    
    # else:
    #     return jsonify({'error': 'Method Not Allowed'}), 405
    


if __name__ == '__main__':
    app.run( debug=True)