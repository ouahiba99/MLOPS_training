# MLOps Engineering Training Program

## Qafza Tech — 12-Week Applied Program

This repository contains my work and projects for the **MLOps Engineering Training Program** at **[Qafza Tech](https://www.qafzatech.com)**.

The goal of this program is to learn how to take a machine learning project from a local development environment and gradually turn it into a **production-ready ML system**.

Throughout the program, I will be working with the **Olist Brazilian E-Commerce Dataset** and building the project step by step, covering machine learning, APIs, Docker, data pipelines, experiment tracking, model serving, monitoring, CI/CD, and infrastructure.

---

## 🎯 What I'm Building

The main project is based on an e-commerce use case:

> **Predict whether an order will be delivered late or on time.**

Instead of only training a model, the goal is to build the complete system around it.

The project will evolve from:

```text
Data → ML Model
```

to:

```text
Data
 ↓
ETL Pipeline
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Experiment Tracking
 ↓
Model Registry
 ↓
API
 ↓
Docker
 ↓
Deployment
 ↓
Monitoring
 ↓
Continuous Retraining
```

---

# 📚 Weekly Roadmap

| Week      | Phase                  | Dates             | Topic                         | What I'll Work On                                                                                 |
| --------- | ---------------------- | ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------------- |
| **1**     | Local Foundations      | 20–27 Jul 2026    | **Leakage-proof ML Pipeline** | Build a clean ML pipeline and make sure there is no data leakage between training and validation. |
| **2**     | Local Foundations      | 27 Jul–3 Aug 2026 | **Deep Learning Pipeline**    | Train a deep learning model with PyTorch and prepare it for inference.                            |
| **3**     | Production APIs        | 3–10 Aug 2026     | **Production API**            | Build a FastAPI service that receives input, validates it with Pydantic, and returns predictions. |
| **4**     | Containerization       | 10–17 Aug 2026    | **Docker**                    | Containerize the ML API and make sure it can run consistently in different environments.          |
| **5**     | Data Pipelines         | 17–24 Aug 2026    | **ETL Pipeline**              | Build an ETL process to extract, clean, transform, and load the data into a database.             |
| **6**     | Data Versioning        | 24–31 Aug 2026    | **DVC**                       | Use DVC to version datasets and models so experiments can be reproduced.                          |
| **7**     | Experiment Tracking    | 31 Aug–7 Sep 2026 | **MLflow**                    | Track experiments, parameters, metrics, artifacts, and model versions using MLflow.               |
| **8**     | Distributed Training   | 7–14 Sep 2026     | **Distributed ML**            | Learn how to scale training and workloads using Ray.                                              |
| **9**     | Feature Store          | 14–21 Sep 2026    | **Feature Management**        | Use Feast to create and manage reusable features for training and inference.                      |
| **10**    | Monitoring             | 21–28 Sep 2026    | **Monitoring**                | Monitor the API, infrastructure, and ML system using Prometheus and Grafana.                      |
| **11**    | Continuous Retraining  | 28 Sep–5 Oct 2026 | **Automation**                | Automate testing, retraining, and deployment using GitHub Actions and Ray Serve.                  |
| **12**    | Infrastructure as Code | 5–12 Oct 2026     | **Terraform**                 | Define and provision the infrastructure using Terraform.                                          |
| **Final** | Capstone               | 12–19 Oct 2026    | **End-to-End ML System**      | Put everything together into one complete MLOps project.                                          |

---

# 🛠️ Technologies

These are the main tools and technologies I will be working with:

**ML / Data**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**API**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

**Containers**

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Data Storage / ETL**

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Python ETL](https://img.shields.io/badge/Python%20ETL-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Versioning / Tracking**

![DVC](https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

**Distributed / Serving**

![Ray](https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white)

**Feature Store**

![Feast](https://img.shields.io/badge/Feast-FF5A00?style=for-the-badge&logoColor=white)

**Monitoring**

![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)

**CI/CD / Infra**

![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-7B42BC?style=for-the-badge&logo=terraform&logoColor=white)

---

# 🛒 Dataset

For this training, I'm using the **Olist Brazilian E-Commerce Dataset**.

The dataset contains information about:

* 👤 Customers
* 🛒 Orders
* 📦 Order items
* 💳 Payments
* ⭐ Reviews
* 🏷️ Products
* 🏬 Sellers
* 🗂️ Product categories
* 📍 Geolocation

I'm using PostgreSQL and Python to work with the data and build the different pipelines required throughout the program.

---

# 🧠 Main Learning Goals

By the end of the program, I want to be comfortable with the full ML lifecycle, not just model training.

Some of the main things I will practice are:

![alt text](<_- visual selection (2).svg>)

---

# 🏗️ Final Project

The final goal is to connect everything together.

![alt text](<graph TD - visual selection.svg>)

The idea is to have a system where new data can go through the pipeline, the model can be retrained when necessary, and the deployed service can be monitored.

---

# 📁 Repository Structure

I'll keep the repository organized by week:

```text
mlops-engineering/
│
├── week-01-ml-pipeline/
├── week-02-deep-learning/
├── week-03-fastapi/
├── week-04-docker/
├── week-05-etl/
├── week-06-dvc/
├── week-07-mlflow/
├── week-08-ray/
├── week-09-feast/
├── week-10-monitoring/
├── week-11-retraining/
├── week-12-terraform/
│
├── capstone/
│
└── README.md
```

Each week will contain the exercises, code, notes, and deliverables related to that topic.

---

# 💻 Prerequisites

The main skills I expect to use throughout the program are:

* Python
* SQL
* Git/GitHub
* Linux command line
* Basic machine learning
* Basic data analysis

I'm also using this program to strengthen my practical knowledge of tools that are commonly used in production ML and data engineering environments.

---

# 📅 Timeline

![alt text](<_- visual selection (1).svg>)

---

## 🚀 Goal

My main goal with this repository is to document my progress while learning **MLOps** and to build a complete project that demonstrates how a machine learning model can move from development to a more realistic production environment.

This repository will be updated throughout the program as I complete each stage.

---

## 📜 License

The training materials provided by Qafza Tech are proprietary and intended for educational purposes.

The code and work in this repository represent my learning and implementation during the program.
