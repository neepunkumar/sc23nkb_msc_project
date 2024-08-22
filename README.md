# sc23nkb_msc_project

SC23NKB_MSC_Project

This repository contains all the resources and code developed as part of the SC23NKB MSc project. The primary focus of this project is the design, development, and evaluation of a custom autoscaler for Kubernetes environments, leveraging advanced machine learning techniques. Below is an overview of the repository structure and the contents of each directory and file.

Repository Structure

1. default-autoscaler
Description: This directory contains the configuration files and scripts necessary for deploying and evaluating the default Kubernetes Horizontal Pod Autoscaler (HPA) in the project. It is used as a baseline to compare the performance against the custom Transformer-based autoscaler developed in this project.
Contents:
HPA configuration YAML files.
Scripts for deploying the default autoscaler in the Kubernetes cluster.
Performance metrics collection scripts.
2. demo video
Description: This directory contains a video demonstration of the project's key features, showcasing the custom autoscaler in action. The demo highlights the deployment process, traffic simulation, and performance monitoring.
Contents:
A video file demonstrating the deployment and scaling processes.
Accompanying scripts or instructions to reproduce the demo environment.
3. kubernetes-custom-autoscaler
Description: This directory contains the main implementation of the custom Transformer-based autoscaler. It includes all the necessary files for setting up and running the autoscaler within a Kubernetes environment.
Contents:
YAML files for deploying the custom autoscaler.
Python scripts for model training and deployment.
Kubernetes deployment configurations for integrating the custom autoscaler with the cluster.
Scripts for collecting performance metrics and logs.
4. teastore-application
Description: This directory includes the deployment files for the TeaStore application, a microservices-based web application used to simulate real-world traffic and workload patterns in the Kubernetes cluster.
Contents:
YAML files for deploying the TeaStore application in Kubernetes.
Configuration files for integrating the application with the custom autoscaler.
Scripts for generating traffic to test the autoscaler.
5. Workload_prediction.ipynb
Description: This Jupyter notebook is used for training and evaluating different machine learning models, including Transformers, LSTM, Bi-LSTM, Random Forest, and SVM, on the web traffic time series dataset. The notebook walks through the data preprocessing, model training, evaluation, and selection of the best model for the custom autoscaler.
Contents:
Data preprocessing steps.
Model training and evaluation code.
Visualizations and metrics for model comparison.
6. merge-csv.com__66b26c77ca945.csv
Description: This file contains the merged and preprocessed dataset used for training the machine learning models. It includes historical web traffic data, which is crucial for the predictive capabilities of the autoscaler.
Contents:
Time series data of HTTP requests with timestamps and traffic counts.
7. README.md
Description: The README file you are currently reading. It provides an overview of the repository, including its structure, contents, and instructions for setup and usage.
Getting Started

Prerequisites
Kubernetes Cluster (on Azure or any other cloud platform).
Docker.
Prometheus and Grafana for monitoring.
Python with dependencies listed in requirements.txt.
Setup Instructions
