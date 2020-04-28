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

- Resampling

# Binary Cross Entropy Loss Function
What is "Binary Cross Entropy Loss Function?"
[BinaryCrossEntropyLoss]

