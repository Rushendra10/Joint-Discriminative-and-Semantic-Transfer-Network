# Pytorch - Joint Discriminative and Semantic Transfer Network (JDSTN)

# Published in IEEE ESDC, 2023:
@INPROCEEDINGS{10149872,
  author={Sidibomma, Rushendra and Sanodiya, Rakesh Kumar},
  booktitle={2023 11th International Symposium on Electronic Systems Devices and Computing (ESDC)}, 
  title={Learning Semantic Representations and Discriminative Features in Unsupervised Domain Adaptation}, 
  year={2023},
  volume={1},
  number={},
  pages={1-6},
  doi={10.1109/ESDC56251.2023.10149872}}

---

# Joint Discriminative and Semantic Transfer Network (JDSTN)

This repository contains the code implementation for the research paper **"Learning Semantic Representations and Discriminative Features in Unsupervised Domain Adaptation"**, presented at IEEE 2023. JDSTN is a novel approach that aligns semantic representations and enhances discriminative features, significantly improving accuracy in unsupervised domain adaptation tasks.

---

## Overview

JDSTN tackles the challenges of domain adaptation by:
- Aligning the semantic representations of source and target domains.
- Promoting discriminative feature learning using pseudo-labels.
- Improving class cluster compactness and separation in feature space.

The model integrates various loss functions to balance semantic alignment and feature discrimination, outperforming existing state-of-the-art methods.

---

## Key Features
- **Semantic Alignment**: Ensures feature similarity across domains for the same class.
- **Discriminative Features**: Reduces intra-class variance and increases inter-class variance for robust classification.
- **Flexible Framework**: Extensible to various domain adaptation scenarios and datasets.

---

## Loss Functions

The model optimizes the following combined loss function:

\[
L = L_c + L_d + L_{\text{sem}} + L_{\text{inter}} + L_{\text{intra}} + L_{\text{ent}}
\]

### 1. **Classification Loss (\(L_c\)):**
Cross-entropy loss on the source domain:
\[
L_c = \frac{1}{n_s} \sum_{i=1}^{n_s} J(f_\theta(x^s_i), y^s_i)
\]

### 2. **Domain Adversarial Loss (\(L_d\)):**
Ensures domain-invariant features using a discriminator:
\[
L_d = \mathbb{E}_{x \sim D_s} [\log(1 - D \circ G(x))] + \mathbb{E}_{x \sim D_t} [\log(D \circ G(x))]
\]

### 3. **Semantic Loss (\(L_{\text{sem}}\)):**
Aligns the centroids of the source and target domain features:
\[
L_{\text{sem}} = \sum_{k=1}^K \Phi(C^k_s, C^k_t)
\]

Where:
\[
C^k_s = \frac{1}{n_s} \sum_{x_i \in X_s} G(x_i), \quad C^k_t = \frac{1}{n_t} \sum_{x_i \in X_t} G(x_i)
\]

### 4. **Inter-Class Variance Loss (\(L_{\text{inter}}\)):**
Maximizes the distance between different class feature clusters:
\[
L_{\text{inter}} = \sum_{i,j=1, y_i \neq y_j} \max(0, m_2 - ||h^s_i - h^s_j||^2)^2
\]

### 5. **Intra-Class Variance Loss (\(L_{\text{intra}}\)):**
Minimizes the distance within the same class feature clusters:
\[
L_{\text{intra}} = \sum_{i,j=1, y_i = y_j} \max(0, ||h^s_i - h^s_j||^2 - m_1)^2
\]

### 6. **Target Conditional Entropy Loss (\(L_{\text{ent}}\)):**
Promotes high-confidence predictions on the target domain:
\[
L_{\text{ent}} = \frac{1}{n_t} \sum_{i=1}^{n_t} \sum_{j=1}^K -p_j \log(p_j)
\]

---

## Architecture

JDSTN is composed of three networks:
1. **Feature Extractor**: Extracts deep features using a pre-trained AlexNet.
2. **Classifier**: Predicts class labels for input data.
3. **Discriminator**: Adversarially trains to enforce domain invariance.

---

## Results

JDSTN achieves state-of-the-art accuracy on the Office-31 benchmark dataset, with substantial improvements over existing methods that use semantic alignment.

