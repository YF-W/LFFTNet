# Layered Feature Fusion and Transformer Integration for Dual Encoding–Mixed Decoding in Semantic Segmentation

![GitHub Repo stars](https://img.shields.io/github/stars/YF-W/LFFTNet)![GitHub Repo forks](https://img.shields.io/github/forks/YF-W/LFFTNet)

Yuefei Wang

Accurate semantic segmentation of medical images is essential for identifying lesions and anatomical structures, supporting both precise clinical diagnosis and intelligent healthcare solutions such as personalized treatment planning and disease monitoring. Traditional encoder–decoder networks, while widely used, often face challenges in maintaining effective feature propagation and integrating contextual information across layers. Moreover, the best strategy for incorporating Transformer-based modules into these architectures has yet to be established. To tackle these issues, we propose a dual-encoder, hybrid-decoder network enhanced with Vision Transformer components. By embedding Transformers in both the encoding and decoding paths, the model captures long-range dependencies and strengthens feature representation. In addition, we introduce a hierarchical feature fusion mechanism that combines information from multiple semantic levels through skip connections, along with a multi-scale resolution fusion module that aligns features across different resolutions to maintain semantic coherence. 

<img width="1306" height="687" alt="image" src="https://github.com/user-attachments/assets/fb471623-d2cf-4540-9b42-6338860aa36f" />


## Main components

### dataset.py

------

The guidelines for data preprocessing.

### LFFTNet.py

------

The specific implementation of the methodology in this paper.

### train.py

------

The details of training method and hyperparameters.

### utils

------

The loader, indicators and their calculation methodology.

### requirements

------

The detailed description of the experimental environment.

## Usage

Run Train.py after preparing the dataset.
