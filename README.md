# Iris Pupil Segmentation

## Introduction 

This challenge aims to estimate the location and diameters of a subject’s pupils and iris from a cropped eye image. The Dataset folder contains a training set of 1000+ images (Close cropped images of the Right and left eyes of subjects) and their associated labels as ellipse coordinates (enter_x, center_y, radius_x, radius_y, theta) and one folder contain a testing set of 100 images and their associated labels.

## Related work 

Prior work such as by C.Wang et al 2021 [1] challenge paper, and the corresponding competition's winners have shown that the segmentation approach with transfer learning would result in models with very good segmentation masks corresponding to the pupil and iris region. This challenge pays more attention to a better segmentation map. Whereas BioTrillion would like to track these landmarks on edge devices, Hence in this work I explore two approaches, 1) Segmentation and 2) Regression approach, where the segmentation approach will look to identify an efficient and smaller backbone and the regression model will predict coordinates of the ellipse corresponding to Iris and Pupil, from just the encoder. 

## Methodology

For this work, I use the "segmentation-models-pytorch", where I use UNet architecture, with different backbones (Resnet18, Resnet34, Resnet50, MobileNet-V2, and Efficientnet-b3). ![](https://i.imgur.com/8dmXn5g.png)
The segmentation network predicts 2 masks separately instead of predicting one single mask. In the case of the regression model, which convolves the output of the last encoder layer, followed by the fc layer. The output of the fc layer predicts the coordinates of the pupil and Iris. 

## Loss Function
### Segmentation Model : 
We have a combination of Dice and BCE that perform well for this task, but apart from that, we have tried other losses such as focal loss.

### Regression Model: 
For the regression model, I tried adding Complete IoU, also called as the Distance IoU, but did not perform well. The loss function that worked well was a combination of L1 Loss and MSE Loss. 

## Dataset Description: 

Image size (PNG image): RGB image (160 X 224 X 3)
Label (dataframe / csv file) : Pupil (center_x, center_y, radius_x, radius_y, theta) and Iris (center_x, center_y, radius_x, radius_y, theta) (1 X 10)

### Data preparation:  

For this work, I would like to precisely draw the boundary of the pupil and iris, hence I pad zeros to the image. Image (160 X 224 X 3) --> (224 X 224 X 3). As resizing of the image might cause a loss of data, hence I stick to the same with just padding additional rows of zeros. The coordinates are normalized between 0 to 1. 

## Results

### Segmentation Model

| Model_Backbone (Parameters)|  Mean % p_y error | Mean % I_y error | Mean % p_IoU | Mean % I_IoU |
| ----------- | -------- | -------- | -------- | -------- | 
| Resnet18 (11M)| 12.71     | 4.38     |75.53     | 90.25     | 
| Resnet34 (21M)| 12.34     | 3.21     |75.02     | 90.84     | 
| Resnet50 (23M)| 11.33    | 3.39     |76.77     | 91.17   | 
| MobileNet-v2 (2M) | 14.03     | 3.51     |75.14     | 90.44     | 
| EfficientNet-b3 (10M)| 100     | 3.96     |0     | 85.42     |

### Regression Model

|  Model_Backbone (Parameters) |  Mean % p_y error | Mean % I_y error | Mean % p_IoU | Mean % I_IoU |
| ----------- | -------- | -------- | -------- | -------- | 
| Resnet18  (11M)   | 13.89     | 3.77     |45.73     | 75.27     | 
| Resnet34  (21M) | 12.14    | 3.46     |46.41     | 75.28     | 
| Resnet50 (23M) | 12.24 | 3.71     |47.03     | 75.45     | 
| MobileNet-v2 (2M)| 16.73     | 5.16     |43.78     | 75.03     | 
| EfficientNet-b3 (10M)| 20.32     | 5.92     |44.73     | 71.99     |

## Code

Wandb Experiment Link : 
1) https://wandb.ai/valudem/Biotrillion_IPSegmentation?workspace=user-valudem
2) https://wandb.ai/valudem/Biotrillion_IPSegmentation_debug?workspace=user-valudem

Google Drive Code : [Code and Weights](https://drive.google.com/drive/folders/1Nufm2vhJV75YsQYhpT5z8JAmZ_iYr2lr?usp=share_link)

1)  The folder contains `colab` folder, which I used to run experiments on Colab. 
2) This code supports wandb logging, is present in the logging folder.
3) Weights_{ModelName}, has the best weights and the prediction saved from the model. The format of the dataframe is (gt_pupil, pred_pupil,gt_iris, pred_iris) 

### Packages:
* numpy
* pandas
* python 3.6
* pytorch 1.5+
* scikit-learn
* scikit-image
* albumentations
* opencv-python
* wandb
* segmentation_models_pytorch


## Reference:

[1] C. Wang et al., "NIR Iris Challenge Evaluation in Non-cooperative Environments: Segmentation and Localization," 2021 IEEE International Joint Conference on Biometrics (IJCB), Shenzhen, China, 2021,