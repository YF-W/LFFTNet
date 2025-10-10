# Layered Feature Fusion and Transformer Integration for Dual Encoding–Mixed Decoding in Semantic Segmentation

![GitHub Repo stars](https://img.shields.io/github/stars/YF-W/LFFTNet)![GitHub Repo forks](https://img.shields.io/github/forks/YF-W/LFFTNet)

Yuefei Wang

Accurate semantic segmentation of medical images is essential for identifying lesions and anatomical structures, supporting both precise clinical diagnosis and intelligent healthcare solutions such as personalized treatment planning and disease monitoring. Traditional encoder–decoder networks, while widely used, often face challenges in maintaining effective feature propagation and integrating contextual information across layers. Moreover, the best strategy for incorporating Transformer-based modules into these architectures has yet to be established. To tackle these issues, we propose a dual-encoder, hybrid-decoder network enhanced with Vision Transformer components. By embedding Transformers in both the encoding and decoding paths, the model captures long-range dependencies and strengthens feature representation. In addition, we introduce a hierarchical feature fusion mechanism that combines information from multiple semantic levels through skip connections, along with a multi-scale resolution fusion module that aligns features across different resolutions to maintain semantic coherence. 

<img width="1306" height="687" alt="image" src="https://github.com/user-attachments/assets/fb471623-d2cf-4540-9b42-6338860aa36f" />


## Main components

### Framework

\begin{table}[!h]
\centering
\caption{Network Architecture Overview}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Network} & \textbf{Layer Name} & \textbf{Branch1 (D(x))} & \textbf{Branch2 (R(x))} & \textbf{Branch3 (V(x))} \\
\hline
\multirow{4}{*}{Encoder} 
& Layer 1 & Conv[3$\times$3, 64]$\times$2 & ResNet34 Layer 1 & - \\
& & D(x1) = [4, 64, 224, 224] & R(x1) = [4, 64, 112, 112] & \\
\cline{2-5}
& Layer 2 & [MaxPool, 64], Conv[3$\times$3, 128]$\times$2 & ResNet34 Layer 2 & \\
& & D(x2) = [4, 128, 112, 112] & R(x2) = [4, 128, 56, 56] & \\
\cline{2-5}
& Layer 3 & [MaxPool, 128], Conv[3$\times$3, 256]$\times$2 & ResNet34 Layer 3 & \\
& & D(x3) = [4, 256, 56, 56] & R(x3) = [4, 256, 28, 28] & \\
\cline{2-5}
& Layer 4 & [MaxPool, 256], Conv[3$\times$3, 256]$\times$2 & ResNet34 Layer 4 & \\
& & D(x4) = [4, 512, 28, 28] & R(x4) = [4, 512, 14, 14] & \\
\hline
Bottleneck & Pre-processing & [MaxPool, 512] & - & \\
& Output Size & D(4x) = [4, 512, 14, 14] & & \\
\hline
FFDSL Layer & F1-F3 & See detailed formulation & & \\
& Output Size & [4, 512, 14, 14] & & \\
\hline
\multirow{4}{*}{Decoder} 
& Layer 4 & DR1 = Conv[3$\times$3, 512](Concat(F3, D(x4), R(x4), V(x1))) & TransConv(Conv[3$\times$3, 256](Concat(DR1, V(x2))) & V(x1) = Vit Layer1, V(x2) = Vit Layer2 \\
& & DR1 = [4, 256, 28, 28] & & \\
\cline{2-5}
& Layer 3 & DR2 = Conv[3$\times$3, 256](Concat(DR1, MaxPool(D(x3)), R(x3), V(x3))) & TransConv(Conv[3$\times$3, 128](Concat(DR2, V(x4))) & V(x3) = Vit Layer3, V(x4) = Vit Layer4 \\
& & DR2 = [4, 128, 56, 56] & & \\
\cline{2-5}
& Layer 2 & DR3 = Conv[3$\times$3, 64](Concat(DR2, MaxPool(D(x2)), R(x2), V(x5))) & TransConv(Conv[3$\times$3, 64](Concat(DR3, V(x6))) & V(x5) = Vit Layer5, V(x6) = Vit Layer6 \\
& & DR3 = [4, 64, 112, 112] & & \\
\cline{2-5}
& Layer 1 & DR4 = Conv[3$\times$3, 64](Concat(DR3, MaxPool(D(x1)), R(x1), V(x7))) & TransConv(Conv[3$\times$3, 64](Concat(DR4, V(x8))), Conv[1$\times$1, 1]) & V(x7) = Vit Layer7, V(x8) = Vit Layer8 \\
& & DR4 = [4, 1, 112, 112] & & \\
\hline
\end{tabular}
\label{tab:network_architecture}
\end{table}


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
