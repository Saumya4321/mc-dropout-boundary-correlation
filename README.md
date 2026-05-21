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
![image](https://github.com/user-attachments/assets/80a07156-4380-4f9f-9c3a-3dd0be9953a5)



## Results
<img width="651" height="205" alt="Screenshot 2025-09-09 121800" src="https://github.com/user-attachments/assets/24ac2f9f-67a9-48e3-bee8-64732bb5a012" />

<br>
<br>
<img width="721" height="814" alt="Screenshot 2025-09-08 171138" align="center" src="https://github.com/user-attachments/assets/e095c65e-fd22-4327-bc18-6b40559920f0" />

### Uncertainty Analysis of Augmentation Techniques in Brain Tumor Segmentation





## Citation

If you find this work useful, please cite:
```
Saumya B, "Uncertainty Analysis of Augmentation Techniques in Brain Tumor Segmentation," 2025.
GitHub repository - https://github.com/Saumya4321/ds261-project
```





