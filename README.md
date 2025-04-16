---
title: "OncoDerm AI"
---

# Project Proposal

### Executive Summary

We propose developing a machine learning-based skin cancer screening system utilizing dermatoscopic images. The system will assist medical professionals in preliminary skin lesion assessment while incorporating robust MLOps practices, responsible AI principles, and an interactive chatbot feature powered by a Large Language Model (LLM) for enhanced user engagement.

### Problem Statement

Skin cancer diagnosis requires expert dermatological knowledge and careful image analysis. While machine learning can assist in this process, deploying such systems in clinical settings requires careful consideration of reliability, explainability, operational excellence, and ease of interaction for medical professionals.

### Dataset

- **Source**: DermaMNIST (based on HAM10000)
- **Classes**: 7 distinct skin lesion categories:
  1. Actinic keratoses and intraepithelial carcinoma (akiec)
  2. Basal cell carcinoma (bcc)
  3. Benign keratosis-like lesions (bkl)
  4. Dermatofibroma (df)
  5. Melanoma (mel)
  6. Melanocytic nevi (nv)
  7. Vascular lesions (vasc)
- **Characteristics**:
  - 28x28 pixel dermatoscopic images
  - 10,015 training images
  - 1,268 validation images
  - 2,239 test images
- **Data Split**: Predefined splits provided by MedMNIST

### Technical Architecture

#### 1. Model Development

- **Base Architecture**:
  - ResNet-18 (modified for 28x28 input)
  - Lightweight models for compatibility with small input size
- **Training Pipeline**:
  - Data augmentation (rotation, flipping)
  - Transfer learning with ImageNet weights
  - Fine-tuning for small images
  - Cross-validation for robust performance estimation

#### 2. MLOps Infrastructure

- **Data Pipeline**:
  - Data ingestion and preprocessing
  - Data versioning
  - Data quality checks
- **Experiment Tracking**:
  - MLflow for model versioning and tracking
  - Hyperparameter optimization
  - Model performance visualization
- **CI/CD Pipeline**:
  - Automated testing (unit, integration, model performance)
  - Automated model deployment
  - Automated documentation generation
- **Monitoring**:
  - Model performance metrics
  - Data drift detection
  - Automatic retraining triggers

#### 3. Production Features

##### Model Robustness & Reliability

- **Explainability**
- **Conformal Predictions**
- **Out-of-Distribution Detection**

##### Clinical Integration

- **Interactive Dashboard**:
  - Real-time inference results
  - Explanations by visualizing integrated gradients
  - Image preprocessing and quality checks
  - Resolution handling and upscaling/downscaling options
- **Conformal Predictions**:
  - Set-valued predictions with guaranteed coverage

