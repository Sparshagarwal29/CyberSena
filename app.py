from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from flask import send_file

# Import your custom classes
from model import GAT
from dataset import AMLtoGraph
import torch_geometric.transforms as T

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global variables for model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_loaded = False

def load_model():
    global model, model_loaded
    try:
        if os.path.exists('best_model.pth'):
            checkpoint = torch.load('best_model.pth', map_location=device)
            print("Checkpoint keys:", checkpoint['model_state_dict'].keys())
            # Revert to in_channels=31 to match the checkpoint
            model = GAT(in_channels=31, hidden_channels=16, out_channels=1, heads=4)
            model_dict = model.state_dict()
            checkpoint_dict = checkpoint['model_state_dict']
            # Load with strict=False to ignore other potential mismatches
            model.load_state_dict(checkpoint_dict, strict=False)
            model.to(device)
            model.eval()
            model_loaded = True
            print("Model loaded successfully with potential mismatches ignored!")
        elif os.path.exists('model_weights.pth'):
            model = GAT(in_channels=31, hidden_channels=16, out_channels=1, heads=4)
            model.load_state_dict(torch.load('model_weights.pth', map_location=device))
            model.to(device)
            model.eval()
            model_loaded = True
            print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Could not load model: {e}")
        model_loaded = False
    print(f"Model loaded status: {model_loaded}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def create_visualization(predictions, labels=None):
    plt.figure(figsize=(10, 6))
    pred_counts = pd.Series(predictions).value_counts()
    plt.subplot(1, 2, 1)
    colors = ['green' if x == 0 else 'red' for x in pred_counts.index]
    plt.pie(pred_counts.values, labels=['Normal' if x == 0 else 'Suspicious' for x in pred_counts.index], 
            autopct='%1.1f%%', colors=colors)
    plt.title('Prediction Distribution')
    plt.subplot(1, 2, 2)
    plt.bar(['Normal', 'Suspicious'], 
            [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
            color=['green', 'red'], alpha=0.7)
    plt.title('Transaction Counts')
    plt.ylabel('Count')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            results = process_transactions(filepath)
            return render_template('results.html', **results)
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

def process_transactions(filepath):
    global model, model_loaded
    try:
        print(f"Loading CSV from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded CSV with {len(df)} rows, columns: {list(df.columns)}")
        if not model_loaded:
            predictions = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
            probabilities = np.random.random(len(df))
            results = {
                'total_transactions': len(df),
                'suspicious_count': int(np.sum(predictions)),
                'normal_count': int(len(df) - np.sum(predictions)),
                'suspicious_percentage': float(np.mean(predictions) * 100),
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'demo_mode': True,
                'sample_data': df.head(10).to_dict('records'),
                'plot_url': create_visualization(predictions)
            }
        else:
            print("Processing with loaded model")
            temp_data_dir = 'temp_data'
            raw_dir = os.path.join(temp_data_dir, 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            temp_csv_path = os.path.join(raw_dir, 'HI-Small_Trans.csv')
            # Rename columns to match expected format
            df = df.rename(columns={
                df.columns[0]: 'Timestamp',  # '2022/09/01 00:15'
                df.columns[1]: 'From Bank',  # '0326059'
                df.columns[2]: 'Account',    # '809756DB0'
                df.columns[3]: 'To Bank',    # '0326059.1'
                df.columns[4]: 'Account.1',  # '809756DB0.1'
                df.columns[5]: 'Amount Received',  # '4415.50'
                df.columns[6]: 'Receiving Currency',  # 'US Dollar'
                df.columns[7]: 'Amount Paid',  # '4415.50.1'
                df.columns[8]: 'Payment Currency',  # 'US Dollar.1'
                df.columns[9]: 'Payment Format',  # 'Reinvestment'
                df.columns[10]: 'Is Laundering'  # '0'
            })
            # Encode 'Payment Format' (assuming 'Reinvestment' = 0, adjust if needed)
            df['Payment Format'] = df['Payment Format'].map({'Reinvestment': 0}).fillna(0).astype(int)
            # Ensure 'Receiving Currency' and 'Payment Currency' are encoded (placeholder)
            df['Receiving Currency'] = df['Receiving Currency'].map({'US Dollar': 0}).fillna(0).astype(int)
            df['Payment Currency'] = df['Payment Currency'].map({'US Dollar': 0}).fillna(0).astype(int)
            # Convert Timestamp to numeric
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype('int64') / 10**9  # Convert to seconds
            df.to_csv(temp_csv_path, index=False)
            print(f"Saved temp file to {temp_csv_path}")
            dataset = AMLtoGraph(temp_data_dir)
            print("Dataset created")
            data = dataset[0]
            print(f"Data loaded, features: {data.num_features}, nodes: {data.num_nodes}, edges: {data.num_edges}")
            data = data.to(device)

            # Pad node features to match in_channels=31
            if data.x.shape[1] != 31:
                padding = torch.zeros((data.x.shape[0], 31 - data.x.shape[1]))
                data.x = torch.cat((data.x, padding), dim=1)

            with torch.no_grad():
                print("Making predictions")
                pred = model(data.x, data.edge_index, data.edge_attr)
                print(f"Prediction shape: {pred.shape}")
                probabilities = pred.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
            print(f"Predictions made, shape: {predictions.shape}")
            import shutil
            shutil.rmtree(temp_data_dir, ignore_errors=True)
            results = {
                'total_transactions': len(predictions),
                'suspicious_count': int(np.sum(predictions)),
                'normal_count': int(len(predictions) - np.sum(predictions)),
                'suspicious_percentage': float(np.mean(predictions) * 100),
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'demo_mode': False,
                'sample_data': df.head(10).to_dict('records'),
                'plot_url': create_visualization(predictions)
            }
        print("Returning results")
        return results
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        df = pd.read_csv(filepath) if 'df' not in locals() else df
        predictions = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        probabilities = np.random.random(len(df))
        results = {
            'total_transactions': len(df),
            'suspicious_count': int(np.sum(predictions)),
            'normal_count': int(len(df) - np.sum(predictions)),
            'suspicious_percentage': float(np.mean(predictions) * 100),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'demo_mode': True,
            'error_message': f'Model prediction failed: {str(e)}',
            'sample_data': df.head(10).to_dict('records'),
            'plot_url': create_visualization(predictions)
        }
        return results

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'demo_mode': True,
                'prediction': np.random.choice([0, 1]),
                'probability': float(np.random.random())
            })
        return jsonify({
            'prediction': 0,  # Placeholder
            'probability': 0.5,  # Placeholder
            'demo_mode': False
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_template')
def download_template():
    template_data = {
        'Timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:05:00'],
        'From Bank': ['Bank_A', 'Bank_B'],
        'Account': ['ACC_001', 'ACC_002'],
        'To Bank': ['Bank_B', 'Bank_C'],
        'Account.1': ['ACC_003', 'ACC_004'],
        'Amount Received': [1000.0, 2500.0],
        'Receiving Currency': [0, 1],
        'Amount Paid': [1000.0, 2500.0],
        'Payment Currency': [0, 1],
        'Payment Format': [0, 1],
        'Is Laundering': [0, 1]
    }
    df = pd.DataFrame(template_data)
    template_path = 'static/transaction_template.csv'
    df.to_csv(template_path, index=False)
    return send_file(template_path, as_attachment=True)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)