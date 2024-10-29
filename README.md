#Custom_Object_Detection_Model
This repository contains a custom object detection model built using PyTorch. The model is inspired by MobileNet and trained on a custom dataset with over 66k+ images featuring a wide variety of objects, including makeup kits, animals, and other miscellaneous items.

Model Overview
The main idea was to implement a custom model that utilises deptwise separable convolutions and train model on single batch size to see if the model can detect object over a wide range of objects, the model has been trained over 66k+ images.
Wheer the custom data was used with interpolated images of size 640 x 640 x 3, the rgb choice of data was to enable model to extract contrasting features but due to limited compute i could not train such big network properly,
However the beginning layers have been observed to extrat some insightful features like separating the object from other non-trained objects.
I have spent 3 weeks trying and observing different close structures with single batch size, this results in violent loss but the performance is quite good.
The object detection model uses a MobileNet-inspired architecture with a focus on optimizing for speed and efficiency. It has been trained to detect certain objects in the dataset, although it's still in its early stages and requires further fine-tuning to achieve higher accuracy across more categories.
I trained these models on a DELL G15 with 16gb ram, 3050 6gb graphic card.
Key Features:
Custom Dataset: Trained on a large dataset containing diverse object types and annotations.
MobileNet-Inspired Architecture: Prioritizes lightweight, efficient inference, making it suitable for edge devices and real-time detection tasks.
Checkpoint-Saving Mechanism: The training loop incorporates a mechanism to save model checkpoints, allowing for resumed training and experimentation with different configurations.
Current Progress
The model successfully detects some objects, but there’s still room for improvement in terms of accuracy and generalization.
Only basic object localization is supported at the moment.
Dataset
The dataset includes a wide range of images, from crowds to animals, with varied and sometimes 'dirty' annotations (e.g., partial object annotations like a human tie without the whole person). The dataset is structured in the COCO format with keys for images and annotations.
dataset- https://www.kaggle.com/c/imagenet-object-localization-challenge
The main model is non multi batched training and "just" works while i also implemented the multi batched training version of model to see if the model is actually capable of doing the task when the graph shows a consistent decline in loss ater 43k images over 60k images range,
while the current model needs more experimentation and computaion as it shows a cear sign of poetntail improvement as it can be observed in graph,
![image](https://github.com/user-attachments/assets/0a37bc7c-f6bf-4784-90b9-7d3257288b90)
while here is the graph for batched training
![image](https://github.com/user-attachments/assets/45834ceb-87ec-4800-8bf0-070b6ae4a5de)

This is an image of how the multi batch training version with first layer at 10th channel is detecting edges ![image](https://github.com/user-attachments/assets/9fb7ac35-e160-409b-b723-35019067a594)
while the 512 channel does some pretty good separations between edges and areas ![image](https://github.com/user-attachments/assets/2e390208-bba9-48f7-8ea6-0f22c08bf840)
the mutli batch model works fine over the animal image data as it was trained on animal images only and here is an example of how it identifies similar object but with confusion
![image](https://github.com/user-attachments/assets/87756922-d32e-4487-96c4-f2106666a68b)
