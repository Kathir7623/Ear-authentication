# Technical Model Report: Ear-Based Biometric Recognition
**Project Syndicate | Deep Learning Architecture for Auricular Identification**

## 1. Executive Summary
This report presents the deep learning model architecture designed for a 2D ear-based biometric recognition system. The model, named **EarNet**, is a Convolutional Neural Network (CNN) optimized for extracting stable, unique embeddings from human ear structures.

## 2. System Architecture
The system follows a sequential pipeline designed for robustness in "in-the-wild" conditions:

1.  **Detection Layer**: Uses Haar Cascades (MCS) to localize the left or right ear region within a full-frame image.
2.  **Preprocessing Layer**: 
    - **CLAHE**: Normalizes lighting across different environments.
    - **Resizing**: Standardizes inputs to 224x224 pixels.
    - **Normalization**: Z-score normalization for neural network stability.
3.  **Feature Extraction (EarNet)**: A 4-block deep CNN that transforms image data into a 128-dimensional embedding.
4.  **Matching Layer**: Calculates Cosine Similarity between query embeddings and the authorized user database.

## 3. EarNet Model Specification
The `EarNet` architecture is designed as follows:

| Layer Type | Configuration | Output Size |
| :--- | :--- | :--- |
| **Input** | RGB Image | 224 x 224 x 3 |
| **Conv Block 1** | 32 filters (3x3), BatchNorm, MaxPool | 112 x 112 x 32 |
| **Conv Block 2** | 64 filters (3x3), BatchNorm, MaxPool | 56 x 56 x 64 |
| **Conv Block 3** | 128 filters (3x3), BatchNorm, MaxPool | 28 x 28 x 128 |
| **Conv Block 4** | 256 filters (3x3), BatchNorm, GlobalAvgPool | 1 x 1 x 256 |
| **Fully Connected** | Linear (256 -> 512), ReLU, Dropout | 512 |
| **Embedding** | Linear (512 -> 128), L2 Normalization | **128** |

## 4. Learning Strategy
The model is trained using **Metric Learning** techniques:
- **Loss Function**: Triplet Loss or Contrastive Loss to minimize distance between matching ears and maximize distance between different identities.
- **Optimizer**: Adam (learning rate = 0.001).
- **Justification**: Unlike standard classification, metric learning allows the system to recognize new users (open-set recognition) without retraining the entire network.

## 5. Biometric Justification
The model focuses on extracting features from five primary auricular landmarks:
1.  **Helix**: The outer rim.
2.  **Anti-helix**: Internal curved ridge.
3.  **Concha**: The hollow part near the canal.
4.  **Tragus**: Small prominence in front of the ear.
5.  **Lobule**: The earlobe.

These features are stable from birth to old age and are highly resistant to facial expression changes, making the `EarNet` model a reliable biometric alternative.

---
*Report generated for Guide Approval | Syndicate Biometric Systems 2026*
