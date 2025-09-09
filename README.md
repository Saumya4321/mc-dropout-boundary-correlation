# Limitations of MC Dropout for Error Localization in 2-D Brain Tumor Segmentation 

## Description
Accurate segmentation of brain tumors in magnetic resonance imaging (MRI) plays a crucial role in diagnosis, treatment planning, and monitoring of disease progression. Deep learning models, particularly convolutional neural networks such as U-Net, have achieved significant success in automating tumor segmentation tasks. However, segmentation models are prone to errors, especially near tumor boundaries where irregular shapes and low contrast make delineation difficult. Uncertainty estimation has been proposed as a means to identify regions where the model’s predictions may be unreliable, with the potential to guide clinicians in reviewing or refining the segmentation output. Monte Carlo (MC) Dropout is a widely used technique for estimating model uncertainty by leveraging dropout layers during inference to generate multiple stochastic forward passes. The resulting variance across predictions is interpreted as model uncertainty.
<br>
<br>
In this project, we investigate the relationship between MC Dropout uncertainty and segmentation errors in the context of 2D brain tumor segmentation. Specifically, we explore whether uncertainty estimates can effectively highlight error-prone regions at tumor boundaries, which are critical areas for clinical decision-making. We further evaluate this relationship under different data augmentation settings to assess the robustness of uncertainty estimates and perform statistical analyses to determine the significance and practical relevance of observed correlations.

## Results

<img width="721" height="814" alt="Screenshot 2025-09-08 171138" text-align="center" src="https://github.com/user-attachments/assets/e095c65e-fd22-4327-bc18-6b40559920f0" />
