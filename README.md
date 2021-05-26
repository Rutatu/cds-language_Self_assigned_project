# cds-language_Self_assigned_project


***Self-assigned project for language analytics class at Aarhus University.***



# Text Classification Using CNN and Pre-trained Glove Word Embeddings: Emotion classification


## About the script


This assignment is a self-assigned project. For this assignment I have chosen to train a Logistic Regression (LR) model and CNN model with pre-trained ```Glove Word Embeddings``` for a text classification task -  classifying tweets based on 4 emotions: joy, sadness, anger and fear. Such task might be useful for multiple purposes. We live in the world where emotional and mental health issues become a big concern. Classifying emotions might be just a first step to test whether, for instance, we can develop clever methods to detect early signs of depression in social media posts or other digitally produced content.

The script trains a Logistic Regression (LR) model on tweets to establish a baseline model performance, it outputs a classification report and a confusion matrix, prints out results to the terminal. Then, it trains a deep learning CNN model using pre-trained ```Glove Word Embeddings``` and outputs classification report anf performance graph, prints out performance results to the terminal.

## Methods

The problem of the task relates to classifying emotions based on tweets. To address this problem, firstly, I have used a 'classical' machine learning solution such as CountVectorization + LogisticRegression to establish a baseline model performance. Afterwards, I have trained a CNN model using pre-trained 100-dimensional  ```Glove Word Embeddings``` . The CNN´s architecture consists of Embedding layer with pretrained ```GloVe ``` weights, Convolutional Layer (CONV) with ReLU activation function, Global Max Pooling layer (GlobalMAXPOOL) and a Fully Connected Layer (FC). The output layer (OUT) uses softmax activation function and has 4 possible classes. 

Model´s architecture: Embedding -> CONV+ReLU -> GlobalMAXPOOL -> FC+ReLU -> OUT(softmax)

CNNs are prone to overfitting, therefore I applied a weight regularization method to CONV and FC layers to minimize the overfitting. I have used L2 regularization to constrain how the model performs (l2 = L2(0.0001)).

More about GloVe algorithm : https://nlp.stanford.edu/projects/glove/
