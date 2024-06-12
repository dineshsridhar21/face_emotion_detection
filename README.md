# FACE_EMOTION_DETECTION

# INTRODUCTION
This project primarily focuses on combining face detection and emotion detection. The emotion detection component utilizes a Convolutional Neural Network (CNN) architecture. A CNN, or ConvNet, is a type of deep learning network that learns directly from data. CNNs are especially effective for identifying patterns in images, enabling them to recognize objects, classify images, and categorize different elements.

# REQUIREMENTS
* Python 
* Keras
* Tensorflow
* Numpy
* Opencv
* You can use an OS(Operating System) like Windows, Linux.

# THE MODEL ARCHITECTURE
  * Convolutional Neural Network (CNN):
  * Conv2D Layers: To extract spatial features from the input images.
  * MaxPooling2D Layers: To reduce the spatial dimensions and computational complexity.
  * Flatten Layer: To convert the 2D matrix data to a vector for the fully connected layers.
  * Dense Layers: Fully connected layers to perform the final classification.
  * Dropout Layers: To prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

# CNN ARCHITECTURE FLOW  
![A-generic-CNN-Architecture](https://github.com/dineshsridhar21/face_emotion_detection/assets/113243447/f7835a8b-07a0-42fe-ace4-5fdc8bd6bec3)

# FACE EMOTION RECOGNITON ARCHITECTURE FLOW
![Architecture for Face Emotion Recignition](https://github.com/dineshsridhar21/face_emotion_detection/assets/113243447/4fd369da-0e83-4df0-aaf7-813e1ec02fa0)

# PROJECT DETAILS
Here is an overview of this project, which focuses on person and emotion detection. The project utilizes two datasets:

 * Emotion Dataset
 * Face Dataset

* The emotion dataset is organized into two directories "Test" and "Train". Each containing seven classes. The images are divided with an 80:20 ratio between training and testing. The Training process employs a CNN architecture, which includes
 *  convolution for feature extraction,
 *  ReLU activation to replace negative values with zero,
 *  pooling to reduce the dimensions by selecting the highest value in each region.
 *  In the final dense layer, all neurons are connected. Upon completion of training and testing, you need save the trained weights for further use. 

In the main file, face encoding is performed to labeled for each person with an single image for an training purpose. After training the face encoding data, the weights are saved as a pickle format and need loaded during runtime. The system captures live images and detects individuals using the BGR (blue, green, red) color format. Additionally, the implementation includes time calculation for emotion detection they have spend on the same, changing the display color from green to red based on the duration for overtime spending on same emotion.

This is a brief overview of face and emotion detection using Python.

# HOW TO USE THE CODE
* First, install all the required packages and set up your IDE. Once the setup is complete, check the URL paths for both datasets and update them according to your directory structure.

* After updating the paths, run the face_emotion_detection.py file. This will generate a pre-trained model weight file named emotion_model_weights.h5.

* Next, open the main.py file.

# SAMPLE RECORDED VIDEO FILE 
https://github.com/dineshsridhar21/face_emotion_detection/assets/113243447/89c3ab16-0bce-4723-a521-5a6352567803

# SUMMARY
This architecture leverages the strengths of CNNs in image processing and integrates them with real-time face recognition capabilities provided by the face_recognition library. This makes it suitable for applications that require both emotion detection and face identification.
