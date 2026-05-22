# An Empirical Study on Variance-based MC Dropout Uncertainty–Error Correlation in 2D Brain Tumor Segmentation

**arXiv:** [2510.15541](https://arxiv.org/abs/2510.15541)


## Overview

Accurate brain tumor segmentation from MRI is vital for diagnosis and treatment planning. Although Monte Carlo (MC) Dropout is widely used for uncertainty estimation, the utility of **variance-based uncertainty** — computed as pixel-wise variance across stochastic forward passes — for identifying segmentation errors, particularly near tumor boundaries, remains insufficiently studied.

This study empirically examines the relationship between variance-based MC Dropout uncertainty and segmentation error in 2D brain tumor MRI segmentation using a U-Net trained under four augmentation settings: none, horizontal flip, rotation, and scaling.

**Key finding:** Variance is a poor proxy for segmentation error in this setting — both globally (r = 0.30–0.38) and especially at boundaries (|r| < 0.05). This suggests that the choice of uncertainty *representation* may matter more than the underlying method itself. Alternative representations such as predictive entropy or mutual information may better capture segmentation errors.

> **Note:** These findings are specific to variance-based uncertainty estimation in this 2D U-Net setup and should not be generalized to MC Dropout broadly.


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


## Dataset

3064 T1-weighted contrast-enhanced MRI images from the [Jun Cheng brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), accessed via [Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation). The dataset includes 233 patients across three tumor types: meningioma (708), glioma (1426), and pituitary (930). Tumor boundaries were manually annotated by three experienced radiologists.


## Method

- **Model:** Vanilla U-Net with dropout layers after each encoder/decoder block (dropout rate = 0.3), trained with Focal Loss to handle class imbalance (~1.66% tumor pixels)
- **Uncertainty estimation:** 50 stochastic forward passes at inference with dropout active; pixel-wise variance across passes used as the uncertainty estimate
- **Correlation analysis:** Pearson and Spearman correlations between per-pixel uncertainty and segmentation error maps, computed per image at both global and boundary levels
- **Augmentation settings:** None (baseline), horizontal flip, rotation (±15°), random scaling (0.8–1.2)
- **Statistical tests:** Paired t-test and Wilcoxon signed-rank test to compare uncertainty–error correlations across augmentation conditions

### Data augmentation settings

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

Global correlations are weak; boundary correlations are negligible. Differences across augmentation settings are statistically significant (p < 0.001) but practically negligible (ΔMean < 0.07).

### Uncertainty heatmaps obtained for different augmentation settings

The figure below shows predicted segmentation masks and variance-based uncertainty maps for the same input image across all four augmentation settings.

<img width="2205" height="1278" alt="uncertainty heatmaps" src="https://github.com/user-attachments/assets/dd0f338e-02c3-41a3-bb77-8a8e921f8cde" />

*Green = predicted mask, Blue = ground truth. Brighter regions indicate higher variance across MC Dropout runs.*

The uncertainty maps show high variance spread across large regions of the image — including areas where the model predicted correctly. The regions of highest uncertainty do not consistently coincide with where the model actually made errors, particularly at tumor boundaries. This visually illustrates the central finding of the study: variance-based MC Dropout uncertainty is not a reliable indicator of where segmentation errors occur.



## Citation

If you find this work useful, please cite:

```
@article{saumya2026mcdropout,
  title={An Empirical Study on Variance-based MC Dropout Uncertainty--Error Correlation in 2D Brain Tumor Segmentation},
  author={Saumya B},
  journal={arXiv preprint arXiv:2510.15541},
  year={2026}
}
```


## Code and Weights

All experimental configurations are available in this repository. Model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1vzOuyyP2BSgmmIgk9kX_5hGxUKkx1DzH?usp=sharing).




