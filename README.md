## Project 4 - Dimension Reduction with PCA and AE Report

## Group Members
- Mauricio Garcia-Paez
    - Wrote code implementing all 3 tasks
    - Recorded video (10 minutes)
- Madelyn Good
    - README and Markdown report
    - Recorded video (5 minutes)

## Project Overview
[Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) is a dimensionality reduction and machine learning method used to simplify a large data set by finding the most important patterns, known as Principal Components.

[Autoencoder (AE)](https://www.kaggle.com/code/residentmario/autoencoders) is a neural network and deep learning method used to compress (encode) and decompress (decode) data (input).

The [MNIST Dataset](https://www.kaggle.com/code/heeraldedhia/mnist-classifier-first-deep-learning-project) contains 70,000 images of handwritten digits: `0` through `9`. Each image is 28x28 pixels.

The goal of this project is to train a model to compress these images using dimensionality techniques like PCA and AE. 

This project is implemented using sklearn, a machine learning library in Python. Each method is extended using PyTorch and MatPlotLib.

## Task 0 - Base Model
![MNIST Test](plots/test.png)
This is an example of the data we will be analyzing. This model is a reconstructed 28x28 image pulled from the MNIST Dataset.

## Task 1 - PCA
### A. Reduce to 2D (for visualization) 
In this task, we apply PCA to simplify the high dimensional MNIST Dataset into just 2 features. The first 2,000 images are plotted.

![PCA 2D Plot](plots/pca-plot.png)

### B. Reduce to 32D (for reconstruction)
Plot

### C. Reconstruct and Compute MSE Loss
After reducing the data we then reconstruct it back to its original size. 

In this task, we compute Mean Squared Error (MSE) to measure how close the reconstructed data is to the original data. 

```text
MSE Loss (32D): 0.017180
```
Low MSE means most of the important information was kept. High MSE means important information was lost.

### D. Compare Image Samples
![PCA Images](plots/pca-compare.png)

Here is a side by side comparison of the original images before compression (left), and the reconstructed images after compression (right). If you can still make out the number in the reconstructed images, that means compression worked well! 


## Task 2 - 1 Layer AE