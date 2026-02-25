# ============================================
# Stage 1: Train the Breast Cancer Model
# ============================================
FROM python:3.10 AS model_training

WORKDIR /app

COPY src/model_training.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt
RUN python model_training.py

# ============================================
# Stage 2: Serve Predictions via Flask
# ============================================
FROM python:3.10 AS serving

WORKDIR /app

# Copy trained model + scaler artifacts from Stage 1
COPY --from=model_training /app/breast_cancer_model.keras /app/
COPY --from=model_training /app/scaler_mean.npy /app/
COPY --from=model_training /app/scaler_scale.npy /app/

# Install dependencies first (better layer caching)
COPY requirements.txt /app/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy application code
COPY src/main.py /app/

# Copy frontend assets — ensure they land at /app/templates and /app/statics
COPY src/templates/ /app/templates/
COPY src/statics/ /app/statics/

EXPOSE 80

CMD ["python", "main.py"]