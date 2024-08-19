from flask import Flask, Response, jsonify
from prometheus_client import start_http_server, generate_latest, Gauge
import logging
import os
import requests
import torch
import torch.nn as nn
from kubernetes import client, config
import threading
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Set up Prometheus metrics
custom_metric = Gauge('custom_metric', 'A custom metric', ['pod_ip'])

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/scale', methods=['POST'])
def scale():
    return jsonify({'status': 'success', 'message': 'Scaling endpoint hit'})

# Check if running inside a Kubernetes cluster
if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'):
    try:
        config.load_incluster_config()
        logging.info("Using in-cluster configuration")
    except config.ConfigException as e:
        logging.error(f"Failed to load in-cluster config: {e}")
else:
    try:
        config.load_kube_config()
        logging.info("Using kube-config file")
    except config.ConfigException as e:
        logging.error(f"Failed to load kube-config file: {e}")

# Create an API client
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()

# Define the Transformer Model class
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, model_dim))
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fc = nn.Linear(model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        src = self.dropout(src)
        tgt = self.dropout(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

# Initialize model parameters
input_dim = 1
model_dim = 64
nhead = 4
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 128
dropout = 0.1

model = TransformerModel(input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
# Load the model if you have a saved state_dict
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

# Function to get metrics from Prometheus
PROMETHEUS_URL = "http://10.0.143.200:80"

def get_metrics(query):
    end_time = int(time.time())
    start_time = end_time - 600  # Last 10 minutes
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params={'query': query, 'start': start_time, 'end': end_time, 'step': '30s'})
    result = response.json()
    if 'data' in result and 'result' in result['data']:
        return result['data']['result']
    return []

def get_active_pods(label_selector):
    try:
        pods = v1.list_namespaced_pod(namespace='default', label_selector=label_selector)
        active_pod_ips = [pod.status.pod_ip for pod in pods.items if pod.status.pod_ip]
        logging.info(f"Active pod IPs retrieved: {active_pod_ips}")
        return active_pod_ips
    except Exception as e:
        logging.error(f"Failed to get active pod IPs: {e}")
        return []

class ProactiveCustomAutoscaler:
    def __init__(self, min_pods, cooldown_time, max_workload_per_pod, removal_strategy_ratio):
        self.min_pods = min_pods
        self.cooldown_time = cooldown_time
        self.max_workload_per_pod = max_workload_per_pod
        self.removal_strategy_ratio = removal_strategy_ratio
        self.current_pods = min_pods
        self.last_scaled_time = time.time()

    def scale(self, predicted_workload):
        current_time = time.time()
        if current_time - self.last_scaled_time < self.cooldown_time:
            return self.current_pods

        required_pods = int(np.ceil(predicted_workload / self.max_workload_per_pod))
        required_pods = max(required_pods, self.min_pods)

        # More aggressive scaling
        if predicted_workload > self.max_workload_per_pod:
            required_pods += 1
        if predicted_workload > 2 * self.max_workload_per_pod:
            required_pods += 1

        if required_pods > self.current_pods:
            self.current_pods = required_pods
        else:
            surplus_pods = (self.current_pods - required_pods) * self.removal_strategy_ratio
            self.current_pods = int(self.current_pods - surplus_pods)
            self.current_pods = max(self.current_pods, self.min_pods)

        self.last_scaled_time = current_time
        return self.current_pods


# Initialize the autoscaler
min_pods = 2
cooldown_time = 60  # 60 seconds
max_workload_per_pod = 50  # max workload one pod can handle in a minute
removal_strategy_ratio = 0.6  # 60% of the surplus pods to be removed

autoscaler = ProactiveCustomAutoscaler(min_pods, cooldown_time, max_workload_per_pod, removal_strategy_ratio)

def scale_deployment(deployment_name, namespace, replicas):
    try:
        scale = apps_v1.read_namespaced_deployment_scale(deployment_name, namespace)
        scale.spec.replicas = replicas
        apps_v1.replace_namespaced_deployment_scale(deployment_name, namespace, scale)
        logging.info(f"Scaled {deployment_name} to {replicas} replicas.")
    except client.exceptions.ApiException as e:
        logging.error(f"Exception when scaling deployment: {e}")

def predict_and_update_metric():
    while True:
        try:
            # Define the query to get total HTTP requests over the last 5 minutes
            http_requests_query = 'sum(sum_over_time(haproxy_backend_http_requests_total{proxy="teastore-backend"}[5m]))'

            # Get the metrics data
            metrics = get_metrics(http_requests_query)
            
            # Log raw metrics data
            logging.info(f"Raw metrics data: {metrics}")

            # Aggregate the request count
            total_request_count = 0
            values = []  # Initialize values to an empty list
            for metric in metrics:
                if 'values' in metric:
                    values = [float(value[1]) for value in metric['values']]
                    total_request_count += sum(values)
                elif 'value' in metric:
                    total_request_count += float(metric['value'][1])
                else:
                    logging.error(f"Unexpected metric format: {metric}")

            logging.info(f"Total request count: {total_request_count}")

            # Ensure the sequence is long enough for the model
            if len(values) >= 10:
                recent_data_sequence = values[-10:]  # Use the last 10 data points
                logging.info(f"Recent data sequence: {recent_data_sequence}")

                # Normalize the input sequence
                recent_data_sequence = np.array(recent_data_sequence)
                min_val = recent_data_sequence.min()
                max_val = recent_data_sequence.max()
                if min_val == max_val:
                    normalized_sequence = recent_data_sequence
                else:
                    normalized_sequence = (recent_data_sequence - min_val) / (max_val - min_val)

                logging.info(f"Normalized data sequence: {normalized_sequence}")

                # Get the prediction from the Transformer model
                sequence = torch.tensor(normalized_sequence).float().unsqueeze(0).unsqueeze(2)
                tgt = sequence.clone()
                with torch.no_grad():
                    prediction = model(sequence, tgt)

                # Take the mean of the predictions if the output is not a single scalar
                predicted_value = prediction.mean().item()
                logging.info(f"Prediction: {predicted_value}")

                # Scale the pods based on the predicted workload
                scaled_pods = autoscaler.scale(predicted_value * 1000)  # Scale predicted value appropriately
                logging.info(f"Scaling to {scaled_pods} pods based on predicted workload {predicted_value}")

                # Scale the Kubernetes deployment
                scale_deployment('teastore-webui', 'default', scaled_pods)
            else:
                logging.warning("Not enough data points for prediction.")

        except Exception as e:
            logging.error(f"Error in prediction and scaling: {e}")
        time.sleep(30)  # Wait for 30 seconds before the next iteration




def start_prometheus_server():
    try:
        start_http_server(8002)  # Change to a different port
        logging.info("Started Prometheus metrics server on port 8003")
    except OSError as e:
        logging.error(f"Failed to start Prometheus server: {e}")

if __name__ == '__main__':
    # Start Prometheus metrics server in a new thread
    threading.Thread(target=start_prometheus_server).start()

    # Start the prediction and metric update loop in a new thread
    threading.Thread(target=predict_and_update_metric).start()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
