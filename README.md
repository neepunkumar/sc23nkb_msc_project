# sc23nkb_msc_project



This repository contains all the resources and code developed as part of the SC23NKB MSc project. The primary focus of this project is the design, development, and evaluation of a custom autoscaler for Kubernetes environments, leveraging advanced machine learning techniques, specifically a Transformer-based model. Below is an overview of the repository structure and the contents of each directory and file.

Repository Structure

1. default-autoscaler
Description: Contains the configuration files necessary for deploying and evaluating the default Kubernetes Horizontal Pod Autoscaler (HPA). These files are used as a baseline to compare against the performance of the custom Transformer-based autoscaler developed in this project.
Contents:
hpa.yaml: Configuration file for deploying the default HPA.
scale-role.yaml: Role configuration for scaling permissions.
scale-rolebinding.yaml: RoleBinding configuration to bind the scale role to the necessary service account.
2. demo video
Description: Includes a video demonstration of the project's key features, showcasing the custom autoscaler in action. The demo highlights the deployment process, traffic simulation, and performance monitoring.
Contents:
MSC_Project-demo.mp4: A video file demonstrating the deployment and scaling processes.
readme.md: Accompanying documentation for the demo video.
### 3. kubernetes-custom-autoscaler
Description: This directory contains the implementation of the custom Transformer-based autoscaler. It includes all the necessary files for building, deploying, and running the autoscaler within a Kubernetes environment.
Contents:
Transformer:
Dockerfile: Dockerfile for creating a container image of the Transformer service.
requirements-base.txt: Base requirements file for the Transformer model.
requirements.txt: Complete list of Python dependencies needed for the Transformer model.
role.yaml: Role configuration for the Transformer service.
rolebinding.yaml: RoleBinding configuration to bind the role to the necessary service account.
scale-role.yaml: Role configuration for scaling permissions.
scale-rolebinding.yaml: RoleBinding configuration to bind the scale role to the Transformer service.
transformer-deployment.yaml: Kubernetes Deployment configuration for deploying the Transformer model.
transformer-service.yaml: Kubernetes Service configuration to expose the Transformer model within the cluster.
transformer_model.pth: Pre-trained Transformer model file used for predictions.
transformer_service.py: Python script to serve the Transformer model as an API within the Kubernetes cluster.
haproxy:
haproxy-configmap.yaml: ConfigMap for HAProxy configurations.
haproxy-deployment.yaml: Deployment configuration for HAProxy, used as a load balancer in the cluster.
haproxy-ingress-clusterrole.yaml: ClusterRole for HAProxy ingress permissions.
haproxy-ingress-clusterrolebinding.yaml: RoleBinding for binding the HAProxy ingress role.
haproxy-service.yaml: Service configuration to expose HAProxy within the cluster.
4. teastore-application
Description: This directory includes the deployment files for the TeaStore application, a microservices-based web application used to simulate real-world traffic and workload patterns in the Kubernetes cluster.
Contents:
teastore-ribbon.yaml: Kubernetes deployment file for the TeaStore application using Ribbon for load balancing.
README.md: Documentation specific to the TeaStore application deployment.
5. Workload_prediction.ipynb
Description: A Jupyter notebook used for training and evaluating various machine learning models (Transformer, LSTM, Bi-LSTM, Random Forest, SVM) on the web traffic time series dataset. This notebook guides through data preprocessing, model training, evaluation, and selection of the best model for the custom autoscaler.
Contents:
Data preprocessing steps.
Model training and evaluation code.
Visualizations and metrics for model comparison.
6. merge-csv.com__66b26c77ca945.csv
Description: The merged and preprocessed dataset used for training the machine learning models. This dataset contains historical web traffic data, which is crucial for developing the predictive capabilities of the custom autoscaler.
Contents:
Time series data of HTTP requests with timestamps and traffic counts.

