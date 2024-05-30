# face_emotion_detection
#introduction
This project which mainly focus on face detection and emotion detection together. This emotion detection architecture is an CNN(Convolutional Neural Network). A convolutional neural network (CNN or ConvNet) is a network architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes, and categories.
#REQUIREMENTS
Python 
Keras
Tensorflow
Numpy
Opencv
You can use an OS(Operating System) Windows, Linux.

The Model Architecture
  Convolutional Neural Network (CNN):
  Conv2D Layers: To extract spatial features from the input images.
  MaxPooling2D Layers: To reduce the spatial dimensions and computational complexity.
  Flatten Layer: To convert the 2D matrix data to a vector for the fully connected layers.
  Dense Layers: Fully connected layers to perform the final classification.
  Dropout Layers: To prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

#Summary
This architecture leverages the strengths of CNNs in image processing and integrates it with real-time face recognition capabilities provided by the face_recognition library, making it suitable for applications requiring both emotion detection and face identification.
