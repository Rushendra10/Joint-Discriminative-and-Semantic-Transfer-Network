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

## Architecture

JDSTN is composed of three networks:
1. **Feature Extractor**: Extracts deep features using a pre-trained AlexNet.
2. **Classifier**: Predicts class labels for input data.
3. **Discriminator**: Adversarially trains to enforce domain invariance.

---

## Results

JDSTN achieves state-of-the-art accuracy on the Office-31 benchmark dataset, with substantial improvements over existing methods that use semantic alignment.

