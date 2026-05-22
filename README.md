# Empirical study on reliability of MC Dropout for Error Localization in 2-D Brain Tumor Segmentation 

## Description
Accurate Segmentation of brain tumors in magnetic resonance imaging (MRI) plays a crucial role in diagnosis, treatment planning, and monitoring of disease progression. But its trust-worthiness is questioned by clinicians along with being extremely challenging near the tumor boundaries. This project investigates the use of Monte Carlo (MC) Dropout to estimate uncertainty and identify segmentation errors in 2D brain tumor segmentation. We analyze whether uncertainty maps effectively highlight error-prone regions, especially at boundaries, and assess the impact of different data augmentations through statistical correlation analysis.


## Contents of Repository

+ ```augmentation_experiments\```\
This directory contains the notebooks for all three data augmentation experiments done and their uncertainty analysis. The different data augmentation techniques are labeled as follows:\
DA1 - Horizontal flip\
DA2 - Rotation\
DA3 - Random scaling

+ ```model_weights\``` \
This directory contains a readme file with the link to download the best performing model weights.

+ ```Brain_Tumor_Segmentation_best.ipynb```\
Notebook containing the best performing segmentation model

+ ```Inference_notebook_for_grading.ipynb```\
This notebook loads the weights of the best performing model and runs inference on the provided test data provided in the zip file. Before running this notebook, download the weights from [link](https://drive.google.com/file/d/1YChnisdNceJbb9c4KcS6WbjdLOr4_B1K/view?usp=sharing) , the test data from ```test_data.zip``` and unzip them.

+ ```test_data.zip```\
This zip file includes 6 images and masks in their respective directories. To be used for evaluation.


## Dataset used

![image](https://github.com/user-attachments/assets/297eaae0-7606-4f4d-a15c-b8d1a2bc2fd6)

For this study, a brain tumor dataset containing 3064 T1-
weighted contrast-enhanced MRI images was used. The data
was collected from Nanfang Hospital and Tianjing Medical
University, China, from 2005 to 2010, by Jun Cheng,
who had
uploaded the entire dataset with its metadata to Figshare. The
dataset consists of T1-weighted contrast-enhanced MRI scans
from 233 patients, and has three kinds of brain tumors - 708
cases of Meningiomas, 1426 gliomas and 930 cases of pituitary
tumors. The tumor border was manually delineated
by three experienced radiologists. A copy of this dataset was
taken from Kaggle, where the scans and corresponding binary
masks were uploaded as 256x256 pixel images.
## Preprocessing workflow
![image](https://github.com/user-attachments/assets/e9bb6d5d-3397-4dde-8e77-542b10a5553a)


## Model architecture
![image](https://github.com/user-attachments/assets/b84f879a-b31f-4a06-86c1-496cc5821b87)



## Experiments conducted
#### Focal loss parameterss
The α and γ parameters
for focal loss were varied to evaluate their impact on segmentation
performance. These experiments were conducted using
the original dataset, without applying any data augmentation
techniques.
Two sets of parameter combinations were tested:
(i) α=0.25, γ=2.0
(ii) α=2.0, γ=0.75

#### Data augmentation

| Technique | % of training dataset applied on | Parameters |
| -------- | -------- | -------- |
| Horizontal Flip | 50% | none |
| Rotation | 50% | Angle: ± 15° |
| Random Scaling | 50% | Range: 0.8 - 1.2 |



## Results
| Augmentation type | Pearson r (Global) | Spearman (Global) | Pearson r (boundary) | Spearman (boundary) |
| --- | --- | --- | --- | --- |
| No augmentation | 0.3365 | 0.1115 | -0.0006 | -0.0054|
| Horizontal Flip | 0.3014 | 0.1198 | 0.0369 | 0.0349 |
| Rotation | 0.3106 | 0.1171 | 0.0458 | 0.0453 |
| Scaling | 0.3781 | 0.1136 | 0.0195 | 0.0171|
### Uncertainty Analysis of Augmentation Techniques in Brain Tumor Segmentation

<img width="2205" height="1278" alt="uncertainty heatmaps" src="https://github.com/user-attachments/assets/dd0f338e-02c3-41a3-bb77-8a8e921f8cde" />




## Citation

If you find this work useful, please cite:
```
Saumya B, "Uncertainty Analysis of Augmentation Techniques in Brain Tumor Segmentation," 2025.
GitHub repository - https://github.com/Saumya4321/ds261-project
```





