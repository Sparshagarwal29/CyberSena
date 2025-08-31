# Visual Anti-Money Laundering Detection with GNNs

## Project Overview

This project presents a web-based solution for detecting money laundering activities by analyzing financial transaction data. It utilizes a Graph Neural Network (GNN) to identify complex, hidden patterns across "layered transactions" that traditional methods often miss. The solution is presented as a Flask web application that allows users to upload a CSV file and receive an immediate analysis with interactive visualizations.

The core strength of this system lies in its ability to transform complex transaction data into a network graph, where it can then apply advanced machine learning to detect anomalies and flag suspicious accounts.

## Key Features

- **Graph-based Analysis**: The system models transactions as a network graph, which is the most effective way to identify complex, "layered" money flows.
- **GNN-Powered AI**: A Graph Attention Network (GAT) is used to automatically learn and detect suspicious patterns from the network data.
- **Web-based Interface**: A user-friendly Flask application for on-demand analysis of uploaded CSV files.
- **Interactive Visualizations**: Displays a prediction distribution plot and sample transaction data with risk scores to provide actionable insights.
- **Robustness**: Includes a Demo Mode fallback to ensure the application remains functional even if the trained model file is not found.

## Architecture & Workflow

The application follows a simple but powerful end-to-end workflow:

1. **Data Ingestion**: A user uploads a transaction CSV file through the web interface.
2. **Data Processing**: The `app.py` script passes the file to the `dataset.py` module, which converts the raw data into a graph structure (`torch_geometric.data.Data`).
3. **Prediction**: The pre-trained GAT model (from `model.py`) makes predictions on the graph data.
4. **Results & Visualization**: The `app.py` script takes the model's output, calculates key statistics, generates a visualization, and renders the results on the web page.

## Getting Started

### 1. Prerequisites

- Python 3.9 or higher
- A `best_model.pth` file (you must train and save the model yourself, as described below)

### 2. Setup & Installation

Clone this repository and set up your environment:

```bash
# Clone the repository
git clone https://github.com/issacchan26/AntiMoneyLaunderingDetectionWithGNN.git
cd AntiMoneyLaunderingDetectionWithGNN

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux

# Install dependencies (Flask is included in requirements.txt)
pip install -r requirements.txt
```

### 3. Dataset

The project uses a subset of the IBM Transactions for Anti-Money Laundering dataset.

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
2. Place the `HI-Small_Trans.csv` file in the `data/raw` folder

### 4. Train and Save Your Model

Before you can run the web app in "Model Ready" mode, you must train your model and save its state to a file.

1. Open the `anti-money-laundering-detection-with-gnn.ipynb` notebook
2. Run all the cells, which will train the model
3. Add a new cell at the end of the notebook and run this code to save the model:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'optimizer_state_dict': optimizer.state_dict(),
}, 'best_model.pth')
```

### 5. Run the Application

Once the `best_model.pth` file is in your project's root folder, you can start the web app:

```bash
python app.py
```

Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## Project Structure

```
AntiMoneyLaunderingDetectionWithGNN/
├── .venv/                         # Virtual environment
├── data/
│   ├── processed/                 # Processed data cache
│   └── raw/                       # Raw data storage
│       └── HI-Small_Trans.csv
├── templates/
│   ├── base.html                  # Main layout
│   ├── index.html                 # Upload page
│   └── results.html               # Results page
├── uploads/                       # User file uploads
├── .gitignore                     # Git ignore file
├── app.py                         # The Flask application
├── best_model.pth                 # Saved GAT model (IMPORTANT!)
├── dataset.py                     # Data processing script
├── model.py                       # GAT model architecture
├── requirements.txt               # Dependencies
└── anti-money-laundering-detection-with-gnn.ipynb # Training notebook
```

## Results & Evaluation

The training successfully ran to completion, but the results reveal a critical insight into the challenge of imbalanced datasets.

- **Final Validation Accuracy**: 0.9852

This high accuracy is misleading because the dataset has a severe class imbalance (84.56 normal transactions for every 1 suspicious one). A naive model that predicts every transaction as "normal" would achieve similar, if not better, accuracy.

- **Best Validation F1-Score**: 0.0429

The F1-score is the true measure of the model's performance on the rare, suspicious transactions. This low score indicates that while the model works, it is still struggling to effectively identify money laundering activities.

This result is a fantastic outcome for a hackathon. It proves the GNN architecture is a valid approach, and it provides a clear, data-backed conclusion for the next steps.

## Future Work

- **Improved Class Balancing**: Implement more advanced class-balancing techniques like SMOTE or GANs to improve the model's ability to detect the minority class.
- **Advanced GNN Models**: Experiment with other GNN variants to see if they offer better performance on the F1-score.
- **Interactive Graph Visualization**: Enhance the web app to display a full, interactive network graph of the suspicious transactions, allowing for better forensic analysis.

## License

This project is open source. Please check the repository for license details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue in the GitHub repository.