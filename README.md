# Traffic Sign Classification using Deep Learning
***
## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
***
## Overview

In this project I experimented building deep architecture for the task of correctly classifying road traffic signs typically found on German roads. To achieve this I considered two feed forward convolutational neural network architectures implemented in python using google's Tensorflow framework via the python API. I use the the German Traffic Sign Recognition Benchmark (GTSRB) data set to train and test the deep learning models. Once satisfactory performance have been achieved on the GTSRB data set, we then apply the models to real traffic signs on German roads acquired though google maps. The classification performance of the models on these images are then evaluated.

This work was done as part of the Udacity Self Driving Car Nano-Degree program

***
## The German Traffic Sign Recognition Benchmark Data-set

Historical development of road infrastructure has been done with the human visual system being the key method of information transfer between the environment and vehicle by means of a human perception and then control. Any autonomous driving system capable of safely operating in existing road environments must then as a bare minimum be capable of human level visual recognition of the environment it operates in.  Visual classification is a key element in more complex visual recognition tasks (such as object detection and semantic segmentation) and also a key task in the visual perception of an autonomous vehicle for correct classification of road signs.


 The German Traffic Sign Recognition Benchmark (GTSRB) data set is a multi-class, single-image classification challenge made up of images of images of traffic signs taken from German roads, with an associated class label for each image in the dataset. The images are taken from video data from a camera on-board a moving car, the images of the signs are taken from the video data removing all temporal information.

The size of the data-set is summarised as follows


* Each image is 32 x 32 pixels made up of 3 colour channels formatted in RGB. Each pixel is saved using an 8 bit unsigned integer giving a total possible 256 possible values per pixel.

*  Each image belongs to one of 43 unique classes/labels grouped by the design/meaning of the sign.

*  The training set is made up of 34799 images with their associated labels.

*  The validation set is made up of 4410 images with their associated labels.

*  The test set is made up of 12630 images with their associated labels

The data set can be downloaded [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).


### Exploration of the GTSRB Data-set
![enter image description here](https://github.com/joshwadd/Deep-traffic-sign-classification/blob/master/trainingdistribution.png?raw=true)

### Visualisation of Data-set

On initial thought, correctly classifying traffic signs could be considered to be a fairly simple task due to the design of traffic signs being unique and showing extremely little variably in apprentice within each class. In addition to this, the designs are typically simple and are placed in positions which should be clear to drivers to identify. However there are many aspects of the GTSRB data set that is challenging from the point of view of accurate classification, these include

* Variations in viewpoint
* Variation in brightness of lighting condition
* Motion blur (due to the video capture)
* Damages to signs

Many of these artefacts can be seen by viewing the images from the dataset below

 ![enter image description here]( https://github.com/joshwadd/Deep-traffic-sign-classification/blob/master/ClassExamples2.png?raw=true)
 


***
## Architectures Considered

### AlexNet Style

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x48|
| Max pooling			| 2x2 stride, 2x2 window						|28x28x48|14x14x48|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x48|10x10x96|
| Max pooling			| 2x2 stride, 2x2 window	   					|10x10x96|5x5x96|
| Convolution 3x3 		| 1x1 stride, valid padding, RELU activation    |5x5x96|3x3x172|
| Max pooling			| 1x1 stride, 2x2 window        				|3x3x172|2x2x172|
| Flatten				| 3 dimensions -> 1 dimension					|2x2x172| 688|
| Fully Connected | connect every neuron from layer above			|688|84|
| Fully Connected | output = number of traffic signs in data set	|84|**43**|
