# Patch-Switching
Patch Switching Based Data Augmentation for Image Classification

Image classification is one of the major fields in computer vision.
Classification in environments that include diverse distortions, such as occlusion and view change, is a challenging task.
In this paper, we propose a data augmentation technique based on patch switching to improve image classification verification performance.
The proposed method generates patches of distorted images which can be used to train image classifiers that are robust to occlusion.
Specifically, training data is augmented by splitting a given image into four parts and moving the split parts randomly within the image.
Evaluation using Tiny-ImageNet200 dataset shows that augmenting data with patch switching results in better performance.


## Developement Environment
Python 3.10.4

cudatoolkit 11.3

PyTorch 11.1.0

cudnn 8.4.1

Tiny-ImageNet200 Dataset

## Data Augmentation by Patch Switching
An image is split into four grid areas, and the split areas are randomly reordered to deform the original image.
The newly created deformed images increase the diversity of the training set.
The original image and the deformed image are paired and used during training.
The attention which was focused on the overall form of the object present in the image is now given to other specific parts of the object as well.

![image](https://github.com/Oh-Jieun/Patch-Switching/assets/105771364/1cddf0ff-36d7-4b08-91e0-7dca6881ceac)

## Results
The model with no augmentation showed an mAP of 11.47%, and the model with the proposed patch switching showed an mAP of 25.27%.
Applying data augmentation with patch switching increased the overall performance 2.2 times.
![image](https://github.com/Oh-Jieun/Patch-Switching/assets/105771364/78644c1c-eb7d-4c68-81cf-427c0dd15885)

