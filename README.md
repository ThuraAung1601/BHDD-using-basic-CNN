# BHDD-using-streamlit

## Burmese Handwritten Digit Dataset
- Dataset Link : https://github.com/baseresearch/BHDD
- Download data.pkl directly in order to get pkl raw file

- Train Images : 60000 with image size (28,28)

![Train Images](Image/trainimgs.png)

- Test Images : 27561 with image size (28,28)

![Test Images](Image/testimg.png)

- Classes : 10, i.e, handwritten digits 0 to 9

![Handwritten1 Images](Image/no1.png)

## BHDD Phase 1 - Basic ConvNet architecture

### Problem Statement
The goal of this project is to create a model that will be able to recognize and determine the handwritten digits from its image by using the concepts of Convolution Neural Network and BHDD dataset. Though the goal is to create a model which can recognize the handwritten digits, it can be extended to letters and an individual’s handwriting. The major goal of the proposed system is understanding Convolutional Neural Network, and applying it to the Burmese handwritten recognition system.

### Install requirements
```{r, engine='bash', count_lines}
tra@thura-pc:~$ pip install -r requirements.txt
```

### Train BHDD with Basic ConvNet Architecture with Dropout
```{r, engine='bash', count_lines}
tra@thura-pc:~$ runipy CNN_train.ipynb
```
- Learning Curves after training with ConvNet
![LearningCurves Images](Image/CnnTrain.png)

### Run and deploy using Streamlit 
```{r, engine='bash', count_lines}
tra@thura-pc:~$ streamlit run app.py
```
### Experiments 
- We tried Single-layer Perceptron, Multi-layer Perceptron and ConvNet.
- The best result - 0.98 F-score of classification is by using ConvNet with Regularization (Dropout). Future works still need to be done for architectural innovation on BHDD.
- Epochs : 15
- GPU Execution
- Tools : OpenCV, Matplotlib, Numpy, Keras, Streamlit
- Results
![Results Images](Image/Table (1).png)

### Demo 
- Try it yourself on [streamlit](https://share.streamlit.io/thuraaung1601/bhdd-using-streamlit/main/app.py)
- Demo GIF
![Demo](Image/Demo.gif)

### Contributors
- Thura Aung
- Khaing Khant Min Paing
- Khant Zwe Naing

### References 
1. https://github.com/baseresearch/BHDD
2. https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
3. Dutt,A, Dutt,A., 2016. Handwritten Digit Recognition Using Deep Learning, International Journal of Advanced Research in Computer Engineering & Technology
