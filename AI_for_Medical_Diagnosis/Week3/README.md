# Week3 of AI for Medical Diagnosis
Welcome to the AI for Medicine Diagnosis Week 3 !!

By the end of this week, you will prepare 3D MRI data, implement an appropriate loss function for image segmentation, and apply a pre-trained U-net model to segment tumor regions in 3D brain MRI images.


In particular, you will:
- Perform image segmentation on 3D MRI data.
- Take random sub-samples from a 3D image.
- Standardize an input image.
- Apply a pre-trained U-Net model.
- Implement a proper loss function for model training (soft dice loss).
- Evaluate model performance by calculating sensitivity and specificity.

<img src="https://miro.medium.com/max/2652/1*eTkBMyqdg9JodNcG_O4-Kw.jpeg" width="100%">

# Explore MRI data
## MRI Data and Image Registration
Magnetic resonance imaging (MRI) is an advanced imaging technique that is used to observe a variety of diseases and parts of the body.

At a high level, MRI works by measuring the radio waves emitting by atoms subjected to a magnetic field. 

<img src="https://miro.medium.com/max/1740/1*yC1Bt3IOzNv8Pp7t1v7F1Q.png">

The MRI scan is one of the most common image modalities that we encounter in the radiology field.
Other data modalities include:
- Computer Tomography (CT),
- Ultrasound
- X-Rays.

Compared to 2D image like X-rays, MRI sequence is a 3D volume.
<p align="center"><img width="75%" src="./images/mri_sequences.jpg"/></p>

The Main disadavantage of processing each MRI slice independently using a 2D segmentation model is
- You lose some context between slices. 

The key idea that we will use to combine the information from different sequences is to **treat them as different channels.**
- Idea : RGB color channel -> Depth channel
- You can extend this idea to stacking more channels than just 3. (But there is a memory limit)
- Challenge : Misalignment problem
- Preprocessing : [image Registration](https://en.wikipedia.org/wiki/Image_registration) is the solution to the misalignment problem.

Note that, most of the 3D volume data in medical setting needs preprocessing step of **image registration**.
<p align="center"><image width="30%" src="images/image_registration.jpg"/></p>

# Image Segmentation
## Segmentation
What is Segmentation?
- The process of defining the boundaries of various tissues.
- The task of determining the class of every point(in 2D : pixel, in 3D : voxel).

Why 3D approach?
- 2D approach loses important **3D context**. 
- For instance, if there is a tumor in one slice, there is likely to be a tumor in the slices right adjacent to it.

Why not use whole sequence of MRI data?
- In the 3D approach, ideally, we'd want to pass in the whole MRI volume into the segmentation model and get out a 3D segmentation map for the whole MRI. However, the size of the MRI volume makes it impossible to pass it in all at once into the model. **It would simply take too much memory and computation.** 

## U-Net
The u-net is convolutional network architecture which consists of **a contracting path followed by an expanding path** for fast and precise segmentation of images.
- blog post by [Heet Sankesara “UNet”](https://towardsdatascience.com/u-net-b229b32b4a71)
- Original Paper : https://arxiv.org/pdf/1505.04597.pdf
- A brief video introduction to U-Net : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## Data Augmentation
Note that in segmentation, you have to apply same geometric transformation to input image and output of the segmentation mask.
<p align="center"><image width="75%" src="images/data_augmentation.jpg"/></p>

Note that, data augmentation should be applied with care so as not to change the data's label.

## Loss function for Segmentation : Soft Dice Loss
[Dice Loss](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) is a popular loss function for segmentation models.
- Works well in the presece of imbalanced data
- In our task of brain tumor segmentation, a very small fraction of the brain will be tumor regions.

The soft dice loss will measure the error between our prediction map, P, and our ground truth map, G.
<p align="center"><image width="50%" src="images/soft_dice_loss.jpg"/></p>

Please check below implementation of **Dice Loss** and **Soft Dice Loss**.

```python
"""Pytorch implementation of Dice Loss"""
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff(probs, targets)
        score = 1 - score.sum() / num
        return score
```

```python
"""python implementation of Soft Dice Loss"""
def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch
```

# Practical Considerations
## Different Populations and Diagnostic Technology
One of the main challenges with applying AI algorithms in the clinic, is achieving reliable generalization.

Examples
- tuberculosis is quite prevalent in India, but unlikely to be as prevalent in the hospitals where we've trained our model in the US.
- Different Resolution of CT scanners
- MRI technology is not standard across the globe and across time. The latest scanners have much higher resolution than older scanners.
- Performance drops when an X-ray classification model is developed on data from US hospitals and is later tested on an external dataset from Latin America.


## External validation
To be able to measure the generalization of a model on a population that it hasn't seen, we want to be able to evaluate on a test set from the new population.
- External validation : when the test set is drawn from the **different** distribution as the training set for the model.
- Internal validation : when the test set is drawn from the **same** distribution as the training set for the model.

And if we find that we're not generalizing to the new population, then we could get a few more samples from the new population to create a small training and validation set and then fine-tune the model on this new data.
- Retrospective(Historical) Data : data which **already have** the disease/condition.
- Real-World / Prospective Data : A prospective study (sometimes called a prospective cohort study) is a type of cohort study, or group study, where participants are enrolled into the study **before** they develop the disease or outcome in question


## Measuring Patient Outcomes
Another challenge for the real-world deployment of AI models is that we need metrics to reflect clinical application.

In the real world, we want to be able to look at the effect of our model on **real patients**.

Approaches
- Decision curve analysis, which can help quantify the net benefit of using a model to guide patient care
- See what happens in the setting of a randomized control trial where we compare patient outcomes for patients on whom the AI algorithm is applied versus those on whom the AI algorithm is not applied. 

- Algorithminc bias : The effect of the model on subgroups of the patients (different ages, sex, and socioeconmic status ..etc).
- Model Interpretation : To understand the inner workings of models to understand how and why they make a certain decision.

References
- https://www.jeremyjordan.me/semantic-segmentation/
