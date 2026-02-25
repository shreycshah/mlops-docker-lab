# Breast Cancer Prediction - MLOps Docker Lab

A containerized ML pipeline that trains a binary classification model on the **Breast Cancer Wisconsin dataset** and serves real-time predictions through a Flask web application. The project demonstrates two Docker deployment patterns: **multi-stage builds** and **Docker Compose with shared volumes**.

---

## Overview

This lab covers the end-to-end workflow of packaging a machine learning application using Docker:

1. **Model Training** — A TensorFlow neural network is trained on four selected features from the Breast Cancer Wisconsin dataset to classify tumors as Benign or Malignant.
2. **Model Serving** — A Flask API loads the trained model and exposes a web interface for users to input cell measurements and receive predictions with confidence scores.
3. **Containerization** — The entire pipeline is containerized using two approaches, each demonstrating a different artifact-sharing mechanism between the training and serving stages.

---

## Original Lab Reference

This project is adapted from **Docker Lab 2** of the MLOps course taught by **Professor Ramin Mohammadi** at **Northeastern University**.

🔗 **Original Lab:** [https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab2)

### Changes from the Original

| Aspect | Original Lab | This Repository |
|---|---|---|
| **Dataset** | Iris flower dataset (3-class) | Breast Cancer Wisconsin (binary) |
| **Features used** | Sepal length, sepal width, petal length, petal width | Mean radius, mean texture, mean smoothness, mean perimeter |
| **Model output** | Softmax (multi-class) | Sigmoid (binary classification) |
| **Preprocessing** | No scaler persistence at inference | StandardScaler parameters saved and loaded at inference |
| **API response** | Predicted class only | Predicted class + confidence score |

---

## Project Structure

```
Lab2_BreastCancer/
├── dockerfile              # Multi-stage Dockerfile (Approach 1)
├── dockerfile.serve        # Serving image used by Docker Compose
├── docker-compose.yml      # Compose file with 2 services (Approach 2)
├── requirements.txt        # Python dependencies
├── src/
│   ├── model_training.py   # Trains and saves the TF model + scaler params
│   ├── main.py             # Flask API for serving predictions
│   ├── templates/
│   │   └── predict.html    # Web interface
│   └── statics/
│       ├── benign.jpg      # Result image for benign prediction
│       └── malign.jpg      # Result image for malignant prediction
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

---

## Approach 1: Multi-Stage Dockerfile

Trains the model at **build time** and packages everything into a single image using `COPY --from`.

```bash
docker build -t breast-cancer-predictor -f dockerfile .
docker run -p 80:80 breast-cancer-predictor
```

Open **http://localhost** in your browser.

---

## Approach 2: Docker Compose

Runs training and serving as **separate containers** at runtime, sharing the trained model via a named Docker volume.

```bash
docker compose up --build
```

Open **http://localhost** in your browser.

To stop and clean up:

```bash
docker compose down -v
```

---

## Useful Commands

```bash
# View running containers
docker ps

# View logs
docker logs bc_serving
docker logs bc_trainer

# Shell into serving container
docker exec -it bc_serving bash

# Remove stopped containers and unused images
docker container prune
docker image prune -a
```

---

## Disclaimer

This project is for **educational purposes only**. It is not intended for clinical diagnosis. Always consult a qualified medical professional for health-related decisions.