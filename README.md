<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/pm25/showlit">
    <img src="/data/dl1.png" alt="Logo" width="640">
  </a>

  <h3 align="center">Project Notebook: Deep Learning with Keras/TensorFlow</h3>

  <p align="center">
    After working on several deep learning projects, I created this notebook to systematically organize my notes for more effective and efficient use in future deep learning projects.
    <br>
    This project notebook covers the practical application of ANN, CNN, and RNN models (including LSTM and GRU) on different types of data, such as image data and sequential data.
    <br />
    <br />
    <a href="https://www.amazon.nl/-/en/Francois-Chollet/dp/1617294438">🌐 Reference 
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Get started with neural networks with 4 examplese">Get started with neural networks with 4 examples</a></li>
    <li><a href="#Strategies to Prevent Overfitting">Strategies to Prevent Overfitting</a></li>
    <li><a href="#Deep Learning for Computer Vision">Deep Learning for Computer Vision</a></li>
    <li><a href="#Deep Learning for Sequential & Text Data">Deep Learning for Sequential & Text Data</a></li>
  </ol>
</details>



## 1️⃣. Get started with neural networks with 4 examples 
**1.1. Classifying Handwritten Digits**
- *Dataset*: MNIST (Image | Multiclass)
- *Key Skills*: Building a Dense network, preprocessing image data (Flatten), softmax activation, categorical crossentropy
  
**1.2. Sentiment Analysis of Movie Reviews**
- *Dataset*: IMDB Reviews (Text | Binary)
- *Key Skills*: Preparing text data (vectorization), sigmoid activation, binary crossentropy, interpreting accuracy and loss
  
**1.3. Categorizing News Articles**
- *Dataset*: Reuters Newswire Topics  (Text | Multiclass)
- *Key Skills*: Multi-label encoding, sparse categorical crossentropy, managing output dimensions.
  
**1.2. Predicting House Prices**
- *Dataset*: Boston Housing (Structured Data | Regression)
- *Key Skills*: Outputting a single continuous value, mse loss, data normalization/standardization, evaluating with MAE.  
---
## 2️⃣.  Strategies to Prevent Overfitting 
**2.1. Getting More (and Better) Training Data**

**2.2. Simplifying architecture (fewer layers/units)**

**2.3. Adding Weight Regularization: L1 (Lasso) and L2 (Ridge) penalties**

**2.2. Dropout**

---
## 3️⃣. Deep Learning for Computer Vision 
**3.1.  Introduction to Convolutional Neural Networks (ConvNets/CNNs)**
- Convolution, Pooling, Feature Hierarchies, Parameter Sharing.
  
**3.2. Building a CNN for MNIST Digits**
- Stacking Conv2D, MaxPooling2D, and Dense layers
  
**3.3. Training a CNN from Scratch on a Small Image Dataset (dog and cat)🚀**
- Organizing directories (train/validation/test), using ImageDataGenerator for flow from directory.
- Choosing appropriate filter sizes and network depth for limited data.
- Data augmentation (rotations, flips, zooms, etc.).

**3.4. Leveraging Pre-trained Models (Transfer Learning)**
- Feature Extraction with a Frozen Base: Using models like VGG16/ResNet as fixed feature extractors.
- Fine-tuning: Unfreezing part of the base model for targeted adaptation to the new dataset.

**3.5. Visualizing what convnets learn**

---
## 4️⃣. Deep Learning for Sequential & Text Data

**4.1 Representing Text: From Words to Vectors**
- Word Embeddings: Embedding layer, learning vs. using pre-trained (GloVe, Word2Vec).

**4.2 Recurrent Neural Networks (RNNs) for Sequences**
- Simple RNNs and their Limitations (Vanishing Gradients).
- Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) Cells.
- Application: IMDB Review Classification using an LSTM layer.

**4.3 1D Convolutions for Sequential Data**
- Building a 1D CNN for the IMDB task.
