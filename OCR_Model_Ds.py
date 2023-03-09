#!/usr/bin/env python
# coding: utf-8

# #Lets get started with Designing model

# import tensorflow

# #TensorFlow is an open-source machine learning framework developed by Google Brain Team that allows developers to create and deploy machine learning models. It was released in 2015 and has since become one of the most popular machine learning libraries. TensorFlow provides a powerful set of tools for building and training different types of machine learning models, including deep neural networks. It supports both CPU and GPU computation and provides a variety of APIs that allow developers to create machine learning models using Python, C++, or other programming languages. TensorFlow also includes a visualization tool called TensorBoard that makes it easy to monitor and analyze the training of machine learning models. TensorFlow is widely used in many applications, including image and speech recognition, natural language processing, and robotics.

# In[1]:


import keras


# #Keras is a high-level neural networks API written in Python and designed to be user-friendly, modular, and extensible. It was developed by François Chollet and was first released in 2015. Keras is designed to simplify the process of building and training deep learning models. It provides a simple and consistent interface that makes it easy to create and customize deep neural networks.
# 
# Keras is built on top of lower-level libraries such as TensorFlow, Theano, or CNTK, and provides an abstracted interface that allows developers to quickly build and experiment with deep learning models. Keras supports various types of neural networks, including convolutional neural networks, recurrent neural networks, and multi-layer perceptrons.
# 
# Keras also includes a range of pre-built models that can be used for tasks such as image classification, text classification, and object detection. Additionally, Keras provides utilities for data preprocessing and augmentation, as well as for visualizing the training process and evaluating model performance.
# 
# Overall, Keras is a powerful and popular tool for building and experimenting with deep learning models, particularly for those who are new to deep learning or do not require the advanced features of lower-level libraries like TensorFlow.

# #OpenCV-Python is a Python library that provides easy access to OpenCV (Open Source Computer Vision) functionalities. OpenCV is an open-source computer vision and machine learning software library that is widely used for a variety of applications, including image and video processing, object detection, face recognition, and more.
# 
# OpenCV-Python is essentially a Python wrapper for the OpenCV C++ library, which allows developers to use OpenCV functionalities in Python programs. It provides easy-to-use APIs for various image and video processing tasks, including loading and saving images and videos, manipulating images and videos, and applying various image processing techniques such as edge detection, image filtering, and more.
# 
# OpenCV-Python is compatible with a wide range of platforms, including Windows, Linux, macOS, and Android. It is also compatible with various Python packages, such as NumPy, SciPy, and Matplotlib, making it easy to integrate with other Python libraries.
# 
# Overall, OpenCV-Python is a powerful and popular library for image and video processing, particularly for those who prefer to work in Python. It offers a rich set of functionalities and can be used for a variety of computer vision and machine learning tasks

# In[2]:


from keras.models import Sequential


# #This line of code imports the Sequential class from the Keras API within the TensorFlow library.
# 
# The Sequential class is used to create a linear stack of layers in a neural network. This means that each layer in the network is connected to the previous layer and the next layer, forming a sequence of layers.
# 
# By importing the Sequential class, you can create an instance of the Sequential model and add layers to it using the add() method, which makes it easier to build a neural network with a sequence of layers.

# In[3]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# #This line of code imports four different layer classes from the Keras API within the TensorFlow library:
# 
# Conv2D: Convolutional layer for 2D spatial convolution over images. This layer is used for feature extraction in image-based tasks like image classification, object detection, and segmentation.
# 
# MaxPooling2D: Max pooling operation for spatial data. This layer is used to reduce the spatial size of the feature maps, which reduces the number of parameters in the model and provides some degree of translation invariance.
# 
# Flatten: Flattens the input, which means that it converts a 2D matrix of features into a 1D vector. This layer is used to connect the convolutional layers to the fully connected layers in the neural network.
# 
# Dense: Fully connected layer, which is a standard neural network layer that connects every neuron in the previous layer to every neuron in the current layer.
# 
# By importing these layer classes, you can use them to create a neural network architecture by instantiating and adding layers to a Sequential model.

# In[4]:


model = Sequential()


# #This line of code creates a new instance of the Sequential model, which is a linear stack of layers in Keras.
# 
# The Sequential model is used to build neural networks by stacking layers on top of each other. You can add layers to the model using the add() method, which adds the layer to the end of the model.
# 
# By creating a new instance of the Sequential model, you can start building your own neural network architecture from scratch or customize an existing architecture by adding, removing, or changing layers.

# In[5]:


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))


# This line of code adds a 2D convolutional layer to the Sequential model.
# 
# Here is what each argument in the Conv2D layer represents:
# 
# 32: The number of filters in the convolutional layer. Each filter is responsible for detecting a specific feature in the input image.
# (3, 3): The size of each filter. In this case, each filter has a width and height of 3 pixels.
# activation='relu': The activation function to be applied to the output of each neuron in the convolutional layer. In this case, the Rectified Linear Unit (ReLU) activation function is used.
# input_shape=(28, 28, 1): The shape of the input to the convolutional layer. In this case, the input is a grayscale image with a width and height of 28 pixels and one channel.
# By adding this convolutional layer to the model, you are creating a feature extractor that will learn to detect features in the input image. The output of this layer will be a set of feature maps that highlight the presence of these features in the input image.

# In[6]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# #This line of code adds a MaxPooling2D layer to the Sequential model.
# 
# Here is what each argument in the MaxPooling2D layer represents:
# 
# pool_size=(2, 2): The size of the pooling window. In this case, a 2x2 window is used to pool the output of the previous convolutional layer.
# Other optional arguments like strides or padding can also be specified in this layer.
# By adding this MaxPooling2D layer to the model, you are reducing the spatial size of the output feature maps generated by the previous convolutional layer. Max pooling is a downsampling operation that extracts the most important features from the feature maps, while discarding the less important ones. This helps to reduce the number of parameters in the model, while preserving the most important features. Max pooling also provides some degree of translation invariance, which means that the model can recognize the same features regardless of their position in the input image

# In[7]:


model.add(Conv2D(64, (3, 3), activation='relu'))


# #This line of code adds another 2D convolutional layer to the Sequential model.
# 
# Here is what each argument in the Conv2D layer represents:
# 
# 64: The number of filters in the convolutional layer. Each filter is responsible for detecting a specific feature in the input image.
# (3, 3): The size of each filter. In this case, each filter has a width and height of 3 pixels.
# activation='relu': The activation function to be applied to the output of each neuron in the convolutional layer. In this case, the Rectified Linear Unit (ReLU) activation function is used.
# By adding this convolutional layer to the model, you are adding another layer to the feature extractor that will learn to detect more complex features in the input image. The output of this layer will be another set of feature maps that highlight the presence of these features in the input image.
# 
# Note that since no input_shape argument is specified, this convolutional layer will take the output feature maps of the previous layer as input.

# In[8]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# This line of code adds another MaxPooling2D layer to the Sequential model.
# 
# Here is what each argument in the MaxPooling2D layer represents:
# 
# pool_size=(2, 2): The size of the pooling window. In this case, a 2x2 window is used to pool the output of the previous convolutional layer.
# By adding this MaxPooling2D layer to the model, you are further reducing the spatial size of the output feature maps generated by the previous convolutional layer. This will help to further reduce the number of parameters in the model and prevent overfitting.
# 
# Note that the previous convolutional layer had 64 filters, so the output of this MaxPooling2D layer will be a set of 64 feature maps that are half the size of the previous set of feature maps.

# In[9]:


model.add(Flatten())


# #This line of code adds a Flatten layer to the Sequential model.
# 
# The Flatten layer is used to flatten the output of the previous layer into a 1D vector, which can then be fed into a fully connected neural network layer.
# 
# By adding this Flatten layer to the model, you are converting the output feature maps of the previous convolutional layers into a flat array of numbers, which can be used as input to a dense neural network layer.
# 
# The number of elements in the flattened output is equal to the number of filters in the previous convolutional layer multiplied by the spatial dimensions of the output feature maps. In this case, the output of the previous convolutional layer had 64 filters, and the spatial dimensions of the output feature maps were reduced to half, so the flattened output will have 64 x (7 x 7) = 3136 elements

# In[10]:


model.add(Dense(128, activation='relu'))


# This line of code adds a Dense layer to the Sequential model.
# 
# Here is what each argument in the Dense layer represents:
# 
# 128: The number of neurons in the dense layer. This is a hyperparameter that you can adjust to change the capacity of the model.
# activation='relu': The activation function to be applied to the output of each neuron in the dense layer. In this case, the Rectified Linear Unit (ReLU) activation function is used.
# By adding this dense layer to the model, you are adding a layer that will learn to combine the flattened output from the previous layer into a set of higher-level features that are more abstract and less tied to the specific spatial locations in the input image. The output of this layer will be a set of 128 activations, which can be used as input to the final output layer of the model.

# In[11]:


model.add(Dense(10, activation='softmax'))


# This line of code adds another Dense layer to the Sequential model.
# 
# Here is what each argument in the Dense layer represents:
# 
# 26: The number of neurons in the dense layer. In this case, there are 26 neurons, one for each possible output class in the model.
# activation='softmax': The activation function to be applied to the output of each neuron in the dense layer. In this case, the Softmax activation function is used. The Softmax function normalizes the outputs of the neurons so that they represent a probability distribution over the possible output classes.
# By adding this dense layer to the model, you are adding the final output layer that will predict the probabilities of each of the 26 possible output classes (assuming that this model is designed to perform a classification task with 26 classes).
# 
# The output of this layer will be a set of 26 probabilities, one for each possible output class in the model. The predicted output class will be the class with the highest probability

# In[12]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# This line of code compiles the Sequential model that was previously defined.
# 
# Here is what each argument in the compile method represents:
# 
# optimizer='adam': The optimization algorithm to be used during training. In this case, the Adam optimizer is used, which is a popular optimizer for deep learning models.
# loss='categorical_crossentropy': The loss function to be optimized during training. In this case, the categorical cross-entropy loss function is used, which is a common loss function for multi-class classification problems.
# metrics=['accuracy']: The metric used to evaluate the performance of the model during training and testing. In this case, the accuracy metric is used, which measures the proportion of correctly classified images.
# By calling the compile method, you are configuring the model for training. The optimizer, loss function, and performance metric are specified so that the model can be trained on the input data and output labels. The compile method does not actually perform any training; it just sets up the model for training.

# Now lets start training the Model=>

# In[13]:


from keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Reshape the data to a 4D tensor
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert the labels to one-hot encoded vectors
from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# #This code is preparing the MNIST dataset for use in training a machine learning model. Here's what each line does:
# 
# from keras.datasets import mnist: This line imports the MNIST dataset from the Keras library.
# (X_train, y_train), (X_test, y_test) = mnist.load_data(): This line loads the MNIST dataset into four NumPy arrays: X_train, y_train, X_test, and y_test. The training data consists of 60,000 28x28 grayscale images of handwritten digits, and the test data consists of 10,000 images.
# X_train = X_train.astype('float32') / 255: This line normalizes the pixel values in the training images by dividing each pixel value by 255. This scales the values down to the range of 0-1, making it easier for the model to learn.
# X_test = X_test.astype('float32') / 255: This line does the same normalization on the test images.
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1): This line reshapes the training data to a 4D tensor with shape (num_samples, height, width, channels). In this case, num_samples is the number of training images, height and width are both 28 (the dimensions of the images), and channels is 1, since the images are grayscale.
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1): This line does the same reshaping on the test data.
# from keras.utils import to_categorical: This line imports the to_categorical function from the Keras utility module.
# y_train = to_categorical(y_train, num_classes=10): This line one-hot encodes the training labels. This means that each label (which represents a digit from 0-9) is converted into a vector of length 10, where all elements are zero except for the one corresponding to the digit. For example, the label 3 would be converted to the vector [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
# y_test = to_categorical(y_test, num_classes=10): This line does the same one-hot encoding on the test labels.

# In[14]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# #This line of code is likely using a machine learning library, such as TensorFlow or Keras, to train a model. Specifically, it's calling the fit method on the model object, which trains the model on the input data (X_train) and target labels (y_train). The validation_data parameter specifies the data to use for validation during training, in this case X_test and y_test.
# 
# The model will be trained for 10 epochs, which means it will iterate over the entire training dataset 10 times. The batch_size parameter specifies the number of samples to use in each batch of training. In this case, the batch size is set to 32, so the model will update its weights after processing 32 samples at a time.
# 
# Overall, this line of code is fitting a machine learning model to the training data and monitoring its performance on the validation set for 10 iterations, using a batch size of 32.

# In[16]:


model.save('model.h5')


# In[ ]:

# This line of code saves a machine learning model to a file named "model.h5" in the current directory using the Keras library.

# The ".h5" extension is used to indicate that the file is in the Hierarchical Data Format (HDF5) format, which is a popular file format for storing large numerical arrays and metadata.

# Once the model is saved, it can be reloaded later using the Keras load_model() function to make predictions on new data without having to retrain the model.

# Here's an example of how to load the saved model:

# from keras.models import load_model

# loaded_model = load_model('model.h5')




