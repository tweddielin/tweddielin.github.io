---
layout: article
title: "Google Street View OCR"
date: 2015-09-15T23:14:02-04:00
modified:
categories: project
excerpt: A Kaggle competition to identify characters from Google Street View Images.
tags: [project]
ads: false
image:
  feature:
  teaser: project_image/2015-09-15-streetocr/chars74k400x400.png
---
#### A Kaggle competition to identify characters from Google Street View Images.

The goal of this <a href="https://www.kaggle.com/c/street-view-getting-started-with-julia"><font color="#3399FF">competition</font></a> on Kaggle is to identify characters from Google Street View images. This dataset is called <a href="http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/"><font color="#3399FF">Chars74K dataset</font></a>, which consists of images of characters selected from Google Street View images. This is also a project done with <a href="http://twchrislin.com/"><font color="#3399FF">Chris</font></a>. He has written a <a href="http://twchrislin.com/project/char74k/"><font color="#3399FF">post</font></a> more about Chars74K dataset and how OCR work.

<figure class="third">
	<img src="/images/project_image/2015-09-15-streetocr/chars74k.jpg" style="width:25%;height:25%;">
	<img src="/images/project_image/2015-09-15-streetocr/chars74k400x400.png" style="width:25%;height:25%;">
	<img src="/images/project_image/2015-09-15-streetocr/kannada.png" style="width:25%;height:25%;">
	<figcaption style="text-align:left">Samples of Chars74K dataset, which consists of images of charasters selected from Google Street View images. Images in this dataset contain English text and Kannada text. The number of classes are 62 in Engilsh and 657 in Kannada.</figcaption>
</figure>

#### Method and Results

In this competition , the kNN ( <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm"><font color="#3399FF">k-Nearest Neighbors algorithm </font></a> ) benchmark accuracy is 40.58% , and the RF ( <a href="https://en.wikipedia.org/wiki/Random_forest"><font color="#3399FF">Random Forest</font></a> ) benchmark accuracy 42.93%. We use the boosting-based algorithm to build our first model and its accuracy is 0.46466, which means we need powerful features to solve this multi-class classification problem. So we follow N. Dalal's HOG method [1] to extract HOG features and then use these feature to train linear SVM model. The accuracy of our this model is 69.70%. The HOG + linear SVM model gains a significant improvement of accuracy from 46.47% to 69.70%, which indicates to get higher accuracy, one should be focus on the feature extraction.

In LeCun, Bengio, Hinton's recent review paper of Deep Learning on Nature[2], it indicates that deep learning allows computational models to learn representations of data with multiple levels of abstraction, i.e. the deep neural network architecture can extract good features automatically, and these methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics. Therefore, we start the first neural network architecture called Multi-Layer Perceptron (MLP).

MLP is a feedforward artificial neural network model, consisting of multiple layers of nodes with each layer fully connected to the next one. We have tried four different models, (1) 1-hidden MLP, (2) 2-hidden MLP, (3) HOG + 1-hidden MLP, (4) HOG + 2-hidden MLP. By comparing model1 & model3 and model2 & model4, one can find that the MLP cannot extract very good features from images, and even using HOG features as input MLP couldn't beat HOG + linear SVM model as shown in table below.

Another architecture in artificial neural network model is Convolutional neural networks (ConvNets). ConvNets are designed to process data that come in the form of multiple arrays, for example a color image composed of three 2D arrays containing pixel intensities in the three colour channels. Many data modalities are in the form of multiple arrays: 1D for signals and sequences, including language; 2D for images or audio spectrograms; and 3D for video or volumetric images. There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers[2]. There are two reasons making ConvNets successful especially in image recognition. First, in array data such as images, local groups of values are often highly correlated, forming distinctive local motifs that are easily detected. Second, the local statistics of images and other signals are invariant to location[2]. Therefore, We build an AlexNet-like ConvNets[3], and get the accuracy up to 75.82%. Then we build another Vgg-like ConvNets[4], and boost the accuracy to 79.29%.

|       Model         |   Accuracy (%)  |
|:-------------------:|:---------------:|
| Benchmark kNN       |     40.58%      |
| Benchmark RF        |     42.93%      |
| GBM model           |     46.47%      |
|HOG + linear SVM     |     69.70%      |
|1-hidden MLP         |     56.86%      |
|2-hidden MLP         |     59.28%      |
| HOG + 1-hidden MLP  |     67.90%      |
| HOG + 2-hidden MLP  |     66.29%      |
|AlexNet-like ConvNets|     75.82%      |
|Vgg-like ConvNets    |     79.29%      |


This problem can be modeled as multi-class classification problem. We build a Vgg-like ConvNet classifier which gain the accuracy at 79.29% and current ranking 3rd/48 (2016/2/27).

<figure>
	<img src="/images/project_image/2015-08-31-cern/kaggle2.png">
</figure>

#### Reference

1. N. Dalal, B. Triggs, “Histograms of oriented gradients for human detection”, IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), page 1:886-893, 2005

2. LeCun Y., Bengio Y., Hinton G., Deep learning. Nature 521, 436–444 (2015).

3. A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012

4. Simonyan, K. and Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556.
