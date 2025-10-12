# Layered Feature Fusion and Transformer Integration for Dual Encoding–Mixed Decoding in Semantic Segmentation

![GitHub Repo stars](https://img.shields.io/github/stars/YF-W/LFFTNet)![GitHub Repo forks](https://img.shields.io/github/forks/YF-W/LFFTNet)

Yuefei Wang

Accurate semantic segmentation of medical images is essential for identifying lesions and anatomical structures, supporting both precise clinical diagnosis and intelligent healthcare solutions such as personalized treatment planning and disease monitoring. Traditional encoder–decoder networks, while widely used, often face challenges in maintaining effective feature propagation and integrating contextual information across layers. Moreover, the best strategy for incorporating Transformer-based modules into these architectures has yet to be established. To tackle these issues, we propose a dual-encoder, hybrid-decoder network enhanced with Vision Transformer components. By embedding Transformers in both the encoding and decoding paths, the model captures long-range dependencies and strengthens feature representation. In addition, we introduce a hierarchical feature fusion mechanism that combines information from multiple semantic levels through skip connections, along with a multi-scale resolution fusion module that aligns features across different resolutions to maintain semantic coherence. 

<img width="1306" height="687" alt="image" src="https://github.com/user-attachments/assets/fb471623-d2cf-4540-9b42-6338860aa36f" />


## Main components

### Framework

| Network | Layer Name | Branch1 (D(x)) | Branch2 (R(x)) | Branch3 (V(x)) |
|---------|------------|----------------|----------------|----------------|
| Encoder | Layer 1 | Conv[3×3, 64] ×2 | ResNet34 Layer 1 | - |
|         | Output Size | D(x1) = [4, 64, 224, 224] | R(x1) = [4, 64, 112, 112] | - |
| Layer 2 | [MaxPool, 64], Conv[3×3, 128] ×2 | ResNet34 Layer 2 | - | - |
|         | Output Size | D(x2) = [4, 128, 112, 112] | R(x2) = [4, 128, 56, 56] | - |
| Layer 3 | [MaxPool, 128], Conv[3×3, 256] ×2 | ResNet34 Layer 3 | - | - |
|         | Output Size | D(x3) = [4, 256, 56, 56] | R(x3) = [4, 256, 28, 28] | - |
| Layer 4 | [MaxPool, 256], Conv[3×3, 256] ×2 | ResNet34 Layer 4 | - | - |
|         | Output Size | D(x4) = [4, 512, 28, 28] | R(x4) = [4, 512, 14, 14] | - |
| Bottleneck | Pre-processing | [MaxPool, 512] | - | - |
|         | Output Size | D(4x) = [4, 512, 14, 14] | - | - |
| FFDSL Layer | F1-F3 | F1 = Concat[D(x4), R(x4)] <br> FT1 = Conv[3×3, 1024](TransConv(F1)) <br> FC1 = Conv[5×5, 1024](Conv[3×3, 1024](F1)) <br> FP1 = Conv[7×7, 1024](MaxPool(F1)) <br> FT2 = Concat(FT1, TransConv(TransConv(FP1))) <br> FC2 = Concat(FC1, TransConv(FP1)) <br> FP2 = Concat(FP1, MaxPool(FT2)) <br> FT3 = Conv[3×3, 1024](FT2) <br> FC3 = Conv[5×5, 1024](FC2) <br> FP3 = Conv[7×7, 1024](FP2) <br> F2 = Concat(MaxPool(FT3), FC3, TransConv(FP3)) <br> F3 = Conv[3×3, 512](F2) | - | - |
|         | Output Size | [4, 512, 14, 14] | - | - |
| Decoder | Layer 4 | DR1 = Conv[3×3, 512](Concat(F3, D(x4), R(x4), V(x1))) <br> TransConv(Conv[3×3, 256](Concat(DR1, V(x2)))) | V(x1) = Vit Layer1 <br> V(x2) = Vit Layer2 | DR1 = [4, 256, 28, 28] |
| Layer 3 | DR2 = Conv[3×3, 256](Concat(DR1, MaxPool(D(x3)), R(x3), V(x3))) <br> TransConv(Conv[3×3, 128](Concat(DR2, V(x4)))) | V(x3) = Vit Layer3 <br> V(x4) = Vit Layer4 | DR2 = [4, 128, 56, 56] |
| Layer 2 | DR3 = Conv[3×3, 64](Concat(DR2, MaxPool(D(x2)), R(x2), V(x5))) <br> TransConv(Conv[3×3, 64](Concat(DR3, V(x6)))) | V(x5) = Vit Layer5 <br> V(x6) = Vit Layer6 | DR3 = [4, 64, 112, 112] |
| Layer 1 | DR4 = Conv[3×3, 64](Concat(DR3, MaxPool(D(x1)), R(x1), V(x7))) <br> TransConv(Conv[3×3, 64](Concat(DR4, V(x8)))) <br> Conv[1×1, 1] | V(x7) = Vit Layer7 <br> V(x8) = Vit Layer8 | DR4 = [4, 1, 112, 112] |


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

# Datasets:
1.	The LUNG dataset:https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
2.	The 2018 Data Science Bowl dataset: https://www.kaggle.com/competitions/data-science-bowl-2018/data
3.	The Skin Lesions dataset: https://www.kaggle.com/datasets/ojaswipandey/skin-lesion-dataset.
4.	The BRIAN MRI dataset:https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
5.	The TOOTH dataset:https://tianchi.aliyun.com/dataset/156596

