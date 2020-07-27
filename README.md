# 4D Longitudinal Segmentation of Brain Tissues and Glioma
![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg) <br>

## Background

This repository is created as part of my Master thesis at TU Delft and Erasmus MC. The goal of this project is to explore whether the temporal information is in favour of improving multi-modality segmentation accuracy and consistency of MRI brain images. The segmenation targets in dataset includes 3 types of normal tissues: White Matter(WM), Grey Matter(GM) and Cerebrospinal fluid (CSF), as well as a typy of tumor: low-grade glioma. Previous work usually concentrates on segmenting individual brain image, but little efforts put on the longitudinal analysis. In this work, we have a series chronological MRI images per patient and try to evaluate the influence of longitudinal information on brain MRI image segmentation in terms of accuracy and consistency. The major contributions of this project are lised below:
* Implemented [3D U-Net](https://arxiv.org/abs/1606.06650), modified 3D version from [2D Res U-Net](https://arxiv.org/abs/1711.10684), modified 3D version from [2D D-resunet](https://ieeexplore-ieee-org.tudelft.idm.oclc.org/document/8898392) and created 3D Direct-connection U-Net to use them as CNN backbones.
* Implemented Pytorch version 3D Convlutional LSTM networks.
* Proposed three types of 4D connection strategies and showed that the connection taking place at the high level feature maps is most effective.
* Analyzed the influence of longtitudinal information on segmentation accuracy based on Dice Coefficient Score(DSC), Hausdorf Distance(HD) and Average Surface Distance(ASD)
* Analyzed the influence of longitudinal information on segmentation consistency on the proposed metrics: Tissue Transformation Rate(TTR) and Tissue Maintaining Rate(TMR) over time. 

The dataset used in this project is provided by Erasmus MC but unavailable to be public due to the confidential issues of patient information. However, the algorithms are available in this repository.

## Architecture of proposed 3D Direct-connection U-Net 
In addition to the original shortcuts in 3D U-Net, a direct concatenation from scaled inputs at each level is added to provide detailed local information. This is manipulation is simple but proven to be effective and better than 3D Res U-Net and 3D Dilation Res U-Net on our dataset.

<div align=center><img width=75% src="/images/directUNet.png"/></div>

## Architectures of proposed 3 types of longitudinal connection strategies listed below(from left to right): backend connection, intermediate connection and shortcur connection. 
Bidirectional Convlutional LSTM networks are placed at different position of U-Net backbone. 

<div align=center>
  <img width=30% src="/images/back.png"/>
  <img width=30% src="/images/center.png"/>
  <img width=30% src="/images/shortcut.png"/>
</div>

## Qualitative Results
### Accuracy comparison between CNN backbones
Although 3D Res U-Net achieves best tumor segmentation DSC, the 3D DC U-Net gives overall improvement in all targets againset 3D U-Net baseline.

<div align=center><img width=75% src="/images/CNNDice_boxplot.png"/></div>

### Comparison between longitudinal connection strategies
I took the 3D U-Net as backbone to test the 3 connection strategies. The results shows that the intermediate connection type outperforms the other two.

<div align=center><img width=75% src="/images/RNNDice_boxplot.png"/></div>

### Accuracy comparison of optimal longitudinal connection type with different CNN backbones
The intermediate-connection strategy with 3D DC U-Net provides best accuracy results.

<div align=center><img width=75% src="/images/longitudinalcomp.png"/></div>

### Influence of longitudinal information on segmentation accuracy
I evaluated the influence of longitudinal information on segmentation accuracy by comparing the result with its CNN backbone. It seems that the temporal information leads to limited improvement.
<div align=center><img width=75% src="/images/DCCNNvsRNN.png"/></div>

### Influence of longitudinal information on segmentation consistency
The results demostrate that longitudinal segmentation provides highest mean TMR, but the transformation stability is not always better than CNN backbone.
<div align=center><img width=75% src="/images/transition_dev_EGD-0125.png"/></div>
<div align=center><img width=75% src="/images/transition_dev_EGD-0505.png"/></div>









