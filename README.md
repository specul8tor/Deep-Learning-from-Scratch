# **Deep Learning Model From Scratch to Classifiy Hand Written Digits**
## *By Azaan Azam*

Created a neural network from scratch, using only numpy instead of tensorflow/keras or pytorch or scikit-learn. The data set used was the MNIST dataset that has 42000 samples
of handwritten digits on a 28x28 grid. 

The way the software is structured is a preprocess.py file that deals with extracting the raw data from the csv files and then turning it
into numpy arrays with the correct dimensions for the models to process. Also includes creating batches.

The layers.py file has all the layers built from scratch in a OOP paradigm. All layers include a forward and backward method that is necessary for optimizing using Sochastic Gradeint
Descent.

The main program resides in main.py and I have already made a prebuilt model and implemented the SGD algorithm for it. My best accuracy is 88%.
