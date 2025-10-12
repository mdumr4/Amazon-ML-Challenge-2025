# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
Our solution leverages a state-of-the-art, tri-modal deep learning model to predict product prices by analyzing textual descriptions, product images, and structured tabular features. The architecture uses a cross-attention mechanism to fuse text and image data, and then intelligently fuses this with tabular features using a **Gated Multimodal Unit (GMU)**. To align with the competition's goals, we train the model on a **log-transformed** price target using a custom, differentiable **SMAPE loss function**, ensuring a robust and highly-optimized solution.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
Initial analysis revealed two key challenges: 1) The data is multi-modal (text, image, tabular), requiring a sophisticated fusion strategy. 2) The target variable, `price`, is extremely right-skewed, which can destabilize training. Our methodology directly addresses both of these challenges.

### 2.2 Solution Strategy
Our approach is a single, end-to-end, tri-modal neural network that learns from all data sources simultaneously.

**Core Innovations:**
1.  **Gated Multimodal Unit (GMU):** Instead of simple concatenation, we use a GMU to dynamically learn the importance of text/image features vs. tabular features for each product, allowing for a more intelligent, context-aware fusion.
2.  **Log-Transformed Target:** We train the model to predict `log(price + 1)` to mitigate the extreme skew of the price distribution and stabilize training.
3.  **Custom SMAPE Loss:** We use a custom `SmoothSMAPELoss` function, a differentiable surrogate for the official competition metric, to ensure our model is directly optimizing for the evaluation criteria.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Our model is composed of three parallel encoders, a two-stage fusion process, and a final regression head:
1.  **Encoders:** A BERT model for text, a ViT for images, and an MLP for tabular data (`value` and `unit`).
2.  **Fusion Stage 1 (Cross-Attention):** Text and image embeddings are fused using a cross-attention mechanism.
3.  **Fusion Stage 2 (GMU):** The fused text/image features are then fused with the tabular features using a Gated Multimodal Unit.
4.  **Regression Head:** The final fused vector is passed to an MLP to predict the log-transformed price.

### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing:** Text from `item_name` and `description` is concatenated and tokenized via `BertTokenizer`.
- **Model Type:** `bert-base-uncased` (fine-tuned).

**Image Processing Pipeline:**
- **Preprocessing:** Images are resized and normalized via `ViTImageProcessor`.
- **Model Type:** `google/vit-base-patch16-224-in21k` (fine-tuned).

**Tabular Processing Pipeline:**
- **Preprocessing:** `value` is normalized; `unit` is converted to an index for an embedding layer.
- **Model Type:** An MLP with an `nn.Embedding` layer for the `unit` feature.

**Fusion and Prediction:**
- **Fusion:** A `GatedMultimodalUnit` (GMU) dynamically combines the projected text/image features with the tabular features.
- **Regression Head:** An MLP that maps the final 128-dimension fused vector to a single log-price value.

---

## 4. Training Strategy

- **Target Variable:** The model is trained to predict the log-transformed price: `log(price + 1)`.
- **Loss Function:** `SmoothSMAPELoss`, a custom differentiable approximation of the official SMAPE evaluation metric.
- **Optimizer:** `AdamW` with a learning rate of `5e-6`.
- **Data Split & Epochs:** The data is split 80/20 for training/validation and trained for 10 epochs.

---

## 5. Model Performance

### 5.1 Validation Results
- **SMAPE Score:** [To be filled after running evaluation and inverse-transforming predictions]
- **Other Metrics:** [To be filled after running evaluation]

---

## 6. Conclusion
Our tri-modal, GMU-based architecture with its custom SMAPE loss and log-transformed target represents a highly-optimized, end-to-end solution tailored specifically to the constraints and evaluation criteria of this competition. This approach is designed to be robust, accurate, and state-of-the-art.

---

## Appendix

### A. Code Artefacts
*The primary code for this solution can be found in the following files:*
- `src/model.py`: Contains the implementation of the `ProductDataset` and the final `CrossModalAttentionModel` with the GMU.
- `src/train.py`: Contains the training script, including the `SmoothSMAPELoss` implementation.