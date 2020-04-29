# Building and Training a Model for Medical Diagnosis

Looking at the Lungs and heart

- Mass: A lung mass is defined as an abnormal spot or area in the lungs that are more than 3 centimeters (cm), about 1 1/2 inches, in size. Spots smaller than 3 cm in diameter are considered lung nodules. The most common causes of a lung mass differ from that of a lung nodule, as well as the chance that the abnormality may be cancer. [[ref](https://www.verywellhealth.com/lung-mass-possible-causes-and-what-to-expect-2249388)]

Even though we do not know above definition of "Lung Mass"


# Training, prediction, and loss
- Training 
During training, an algorithm is shown images of chest X-rays labeled with whether they contain a mass or not
- Prediction
The algorithm produces an output in the form of scores, which are probabilities that the image contains a mass.
- Loss
From the probability score that the model predicted, we compute "Error" with the desired score.

# Image Classfication and Class Imbalance
Three Key Challenges
- Class Imbalance 
- Multi-Task
Typical Solution : Multi-Label Loss 
- Dataset Size
Typical Solution : Transfer Learning + Data Augmentation

## Class Imbalance Problem
What is Class Imbalance problem?
In a medical dataset, it's common to have not an equal number of examples of non-disease and disease.
- normal case > abnormal case 

[ClassImbalance]

Common Approches to solve "Class Imbalance" is 
- Weighted Loss : By counting the number of each labels and modifying the loss function to weighted loss with the ratio of each label 
[WeightedLoss]

- Resampling : Re-sample the dataset such that we have an equal number of normal and abnormal examples

You can use just standard loss function (not a weighted loss function)

cons
- may not be able to include all of the normal examples in re-sample data - may have more than one copy of abnormal examples which may lead to overfitting to the example

There are many variations of Resampling
- Oversampling the normal/abnormal case
- Undersampling the normal/abnormal case

# Binary Cross Entropy Loss Function
What is "Binary Cross Entropy Loss Function?"
[BinaryCrossEntropyLoss]


# Multi-task challenge -> Multitask learning
Real World Problem is usually not a binary classficiation, but a  Multi-Task.
- Mass or No Mass
- Pneumonia or No Pneumonia
- Edema or No Edema

We define **"Multi-Label/Multi-Task Loss"**.
For Multi-Task learning, We can apply the "weighted loss" that we have covered earlier.


# Dataset size : Working with a Small Training Set
"Convolutional Neural Network" is the most common and well suited architecture for processing image which require millions of examples in image classification.

However, the common dataset size in medical imaging is about 10 thousand to 100 thousand.

1. Pretrain the Network
2. Fine Tuning

Principle of Transfer Learning 
- the early layers of the network : Low level image features / Broadly generalizable / Edges of image
- the later layers of the network : High level image features / More specific to the task 

How to "Transfer Learning"?
case 1. to fine tune all of the layers
case 2. freeze early layers and only fine-tune the later or the last layer

Generating More Samples : Data Augmentation
- Flipping
- Rotation
- Translation
- Zoom
- Change brightness or contrast
- Random Cropping
- Noise Insertion
- ...

Things to Keep in Mind when applying Data Augmentation
1. Does the transformation will make the network generalize better?
2. Do Augmentation Keep the Label the Same?
