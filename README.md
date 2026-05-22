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
# An Empirical Study on Variance-based MC Dropout Uncertainty–Error Correlation in 2D Brain Tumor Segmentation

**arXiv:** [2510.15541](https://arxiv.org/abs/2510.15541)

---

## Overview

Accurate brain tumor segmentation from MRI is vital for diagnosis and treatment planning. Although Monte Carlo (MC) Dropout is widely used for uncertainty estimation, the utility of **variance-based uncertainty** — computed as pixel-wise variance across stochastic forward passes — for identifying segmentation errors, particularly near tumor boundaries, remains insufficiently studied.

This study empirically examines the relationship between variance-based MC Dropout uncertainty and segmentation error in 2D brain tumor MRI segmentation using a U-Net trained under four augmentation settings: none, horizontal flip, rotation, and scaling.

**Key finding:** Variance is a poor proxy for segmentation error in this setting — both globally (r = 0.30–0.38) and especially at boundaries (|r| < 0.05). This suggests that the choice of uncertainty *representation* may matter more than the underlying method itself. Alternative representations such as predictive entropy or mutual information may better capture segmentation errors.

> **Note:** These findings are specific to variance-based uncertainty estimation in this 2D U-Net setup and should not be generalized to MC Dropout broadly.

---

## Repository Structure

```
mcd-error-correlation/
├── segmentation_notebooks/     # U-Net training notebooks for each augmentation setting
├── uncertainty-err/            # Uncertainty estimation and correlation analysis notebooks
├── helper_code_n_scripts/      # Utility scripts
├── data/                       # Dataset-related files
├── model_weights/              # README with link to download trained weights
├── figures/
│   └── training_curves/        # Training curve plots for each run
└── results/                    # Saved correlation values and uncertainty metrics
```

---

## Dataset

3064 T1-weighted contrast-enhanced MRI images from the [Jun Cheng brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), accessed via [Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation). The dataset includes 233 patients across three tumor types: meningioma (708), glioma (1426), and pituitary (930). Tumor boundaries were manually annotated by three experienced radiologists.

---

## Method

- **Model:** Vanilla U-Net with dropout layers after each encoder/decoder block (dropout rate = 0.3), trained with Focal Loss to handle class imbalance (~1.66% tumor pixels)
- **Uncertainty estimation:** 50 stochastic forward passes at inference with dropout active; pixel-wise variance across passes used as the uncertainty estimate
- **Correlation analysis:** Pearson and Spearman correlations between per-pixel uncertainty and segmentation error maps, computed per image at both global and boundary levels
- **Augmentation settings:** None (baseline), horizontal flip, rotation (±15°), random scaling (0.8–1.2)
- **Statistical tests:** Paired t-test and Wilcoxon signed-rank test to compare uncertainty–error correlations across augmentation conditions

---

## Results

| Augmentation | Pearson r (Global) | Pearson r (Boundary) |
|---|---|---|
| No augmentation | 0.3365 | -0.0006 |
| Horizontal flip | 0.3014 | 0.0369 |
| Rotation | 0.3106 | 0.0458 |
| Scaling | 0.3781 | 0.0195 |

Global correlations are weak; boundary correlations are negligible. Differences across augmentation settings are statistically significant (p < 0.001) but practically negligible (ΔMean < 0.05).

---

## Understanding the Statistical Tests

### What is a p-value?

A p-value answers: *if there were actually no difference between two conditions, how likely would it be to see a result as extreme as ours just by chance?*

- p = 0.05 means there is a 5% chance the result happened randomly
- p < 0.001 means it is extremely unlikely to be random noise

**Important:** a small p-value only tells you the difference is *real*, not that it is *large* or *meaningful*. This is why the paper reports both statistical significance (p < 0.001) and practical significance (ΔMean < 0.05) — the differences between augmentation settings are detectable but negligibly small. Large sample sizes give tests enormous statistical power, which is why even tiny differences yield very small p-values.

### What is a Paired t-test?

Imagine you want to know if a drug reduces blood pressure. You measure 10 patients *before* and *after* taking the drug. Each patient gives you a pair of values. The paired t-test looks at the *difference within each patient* — did their own blood pressure go down? — rather than comparing two independent groups. This accounts for natural variation between individuals.

In this study, each test image plays the role of a "patient." It gets one Pearson correlation value under no-augmentation and one under (say) horizontal flip. The paired t-test asks: within each image, did the uncertainty–error correlation change between conditions? It then checks whether that change is consistent enough across all images to be non-random.

The Wilcoxon signed-rank test serves as a non-parametric backup — it makes no assumption about the distribution of differences, making the conclusions more robust.

---

## Citation

If you find this work useful, please cite:

```
@article{saumya2025mcdropout,
  title={An Empirical Study on Variance-based MC Dropout Uncertainty--Error Correlation in 2D Brain Tumor Segmentation},
  author={Saumya B},
  journal={arXiv preprint arXiv:2510.15541},
  year={2025}
}
```

---

## Code and Weights

All experimental configurations are available in this repository. Model weights can be downloaded from [Google Drive](https://drive.google.com/file/d/1YChnisdNceJbb9c4KcS6WbjdLOr4_B1K/view?usp=sharing).




