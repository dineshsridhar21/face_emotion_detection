# FACE_EMOTION_DETECTION

# INTRODUCTION
This project which mainly focus on face detection and emotion detection together. This emotion detection architecture is an CNN(Convolutional Neural Network). A convolutional neural network (CNN or ConvNet) is a network architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes, and categories.

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

# HOW TO USE THE CODE
First install all the requeriments and IDE. Once you setup has been finished check the url path for the both file and change the as per your url loacted. Once done run the file face_emotion_detection.ipynb file. once run the file you will get an pre-trained model weight named as an "emotion_model_weights.h5". Next you can run the main.ipynb file and as per i mention change the file path [directory="....."] and [emotion_model="....."] on the main.ipynb file once you done eith changing the url of directory file run the main.ipynb file on your IDE  

# SUMMARY
This architecture leverages the strengths of CNNs in image processing and integrates it with real-time face recognition capabilities provided by the face_recognition library, making it suitable for applications requiring both emotion detection and face identification.
