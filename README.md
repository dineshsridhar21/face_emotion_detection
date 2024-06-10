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

# PROJECT DETAILS
* The overview of this project which mainly focus in person and emotion detection. I have an some set of dataset they are

* Two Dataset
  * Emotion Dataset 
  * Face Dataset

* In this emotion dataset, I have been included two directory named as "Test" and "Train". In that particular, it will contain 7 classes for both "test" and "train". The  image of 80:20 ratio for train and test. In this training i will do cnn architecture proccess it bascially filtering the process by Convolutional, replace the negative value by the 0 of relu and pooling was to done reduce the layer by the highest value. In the last layer of dense layer it will connect all the neurons. If all the training and testing has done, then you will be able to save the "weights" which you have been train.

* In the main file, You will encoding for face and then know the face of their own label. once you trained the data of the face encoding need to dump as pickle weights and load the weight files while running. It will capture the live image and detect the person with BGR(blue, green, red). Additionaly i have implemented as an time calculation spend on the same emotion, it will change green to red by the timing accourdingly.

* This is the short brief note of face emotion detection using an python. 

# HOW TO USE THE CODE
First install all the requeriments and IDE. Once you setup has been finished check the url path for the both file and change the as per your url loacted. Once done run the file face_emotion_detection.ipynb file. once run the file you will get an pre-trained model weight named as an "emotion_model_weights.h5". Next you can run the main.ipynb file and as per i mention change the file path [directory="....."] and [emotion_model="....."] on the main.ipynb file once you done eith changing the url of directory file run the main.ipynb file on your IDE  

# SUMMARY
This architecture leverages the strengths of CNNs in image processing and integrates it with real-time face recognition capabilities provided by the face_recognition library, making it suitable for applications requiring both emotion detection and face identification.
