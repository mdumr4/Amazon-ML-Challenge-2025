# ML Challenge 2025: Final Solution - A Two-Model Ensemble

**Team Name:** AML
**Team Members:** Mohammed Umar F, Soujanya K Hegde, Soha Rida Khan, Akshay Shanbhag
**Submission Date:** 13 October 2025

---

## 1. Executive Summary
Our final solution is a powerful **two-model ensemble** that leverages the unique strengths of different model architectures to achieve a highly accurate and robust price prediction. The ensemble combines a state-of-the-art, tri-modal deep learning network with a fast and efficient LightGBM model. The final prediction is an average of the outputs from these two "expert" models, a technique proven to reduce error and improve generalization.

---

## 2. Methodology Overview

### 2.1 Solution Strategy
Our strategy is based on the principle that different types of models learn different patterns in the data. By combining their "opinions," we can achieve a result that is superior to any single model.

- **Model 1: The "Fusion Powerhouse"**: A deep neural network designed to understand the complex, unstructured data (text and images) and the interactions between all modalities.
- **Model 2: The "Metadata Expert"**: A LightGBM model, which is a type of gradient boosting machine that excels at finding patterns in structured, tabular data.
- **Ensembling**: The final prediction is a simple average of the outputs from these two models, providing a balanced and more accurate result.

### 2.2 Core Innovations

1.  **Hybrid Architecture:** Combining a deep learning model with a gradient boosting model captures a wider variety of signals in the data.
2.  **Advanced Preprocessing:** We employ a detailed preprocessing pipeline including text cleaning, image augmentation (for the neural network), Z-score normalization, and entity embeddings for categorical features.
3.  **Log-Transformed Target & Custom Loss:** We train on the log-transformed price and use a custom `SmoothSMAPELoss` to directly optimize for the competition's evaluation metric.

---

## 3. Model Architectures

### 3.1 Model 1: The Fusion Powerhouse (Tri-Modal NN)

This is a complex neural network with three parallel encoders whose outputs are intelligently fused.

- **Text Encoder:** A `bert-base-uncased` model processes the cleaned `item_name`.
- **Image Encoder:** A `facebook/dinov2-base` Vision Transformer processes the product images. Training images are augmented with random flips, rotations, and color jitter.
- **Tabular Encoder:** An MLP processes the Z-score normalized `pack_size` and `quantity_value`, along with an embedding for the `quantity_unit`.
- **Fusion:** Text and image features are fused with cross-attention, and the result is then fused with the tabular features using a Gated Multimodal Unit (GMU).
- **Output:** A final regression head predicts the log-transformed price.

### 3.2 Model 2: The Metadata Expert (LightGBM)

- **Model Type:** `lightgbm.LGBMRegressor`.
- **Inputs:** Trained exclusively on the structured tabular features: `pack_size`, `quantity_value`, and `quantity_unit` (which is label-encoded).
- **Training:** The model is trained to predict the log-transformed price and uses early stopping based on validation performance to find the optimal number of boosting rounds.

---

## 4. Training & Prediction

### 4.1 Training

1.  The Fusion Powerhouse (NN) is trained for 10 epochs using the `train.py` script. The best performing model is saved as `best.pth`.
2.  The Metadata Expert (LGBM) is trained using the `train_lightgbm.py` script, which saves the final model as `lightgbm_model.pkl`.

### 4.2 Prediction

The `predict.py` script orchestrates the final prediction:
1.  It loads both the trained neural network (`best.pth`) and the trained LightGBM model (`lightgbm_model.pkl`).
2.  It generates two sets of log-price predictions for the test set, one from each model.
3.  It averages these two sets of predictions.
4.  It applies the inverse log transform (`expm1`) and a positive price safeguard to the final averaged predictions.
5.  It saves the result to `test_out.csv` in the required submission format.

---

## 5. Appendix: Code Artefacts

- **`src/model.py`**: Defines the architecture for the tri-modal neural network.
- **`src/train.py`**: Trains the tri-modal neural network.
- **`src/train_lightgbm.py`**: Trains the LightGBM tabular model.
- **`src/predict.py`**: Loads both models and generates the final ensembled prediction.
- **`src/evaluate.py`**: A utility to perform in-depth analysis on a trained model's performance on the validation set.