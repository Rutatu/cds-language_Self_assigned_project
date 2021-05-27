# cds-language_Self_assigned_project


***Self-assigned project for language analytics class at Aarhus University.***



# Text Classification Using CNN and Pre-trained Glove Word Embeddings: tweets classification based on emotions


## About the script

This assignment is a self-assigned project. For this assignment I have chosen to train a Logistic Regression (LR) model and a Convolutional Neural Network (CNN) model with pre-trained ```Glove Word Embeddings``` for a text classification task -  classifying tweets based on 4 emotions: joy, sadness, anger and fear. Such task might be useful for multiple purposes. We live in the world where emotional and mental health issues became a big concern. Classifying emotions might be just a first step to test whether, for instance, we can develop clever methods to detect early signs of depression in social media posts or other digitally produced content.

The script trains a LR model on tweets to establish a baseline model performance, it outputs a classification report and a confusion matrix, prints out results to the terminal. Then, it trains a deep learning CNN model using pre-trained ```Glove Word Embeddings``` and outputs classification report anf performance graph, prints out performance results to the terminal.

## Methods

The problem of the task relates to classifying emotions based on tweets. To address this problem, firstly, I have used a 'classical' machine learning solution such as CountVectorization + LogisticRegression to establish a baseline model performance. Afterwards, I have trained a CNN model using pre-trained 100-dimensional  ```Glove Word Embeddings```. For the first run the weight were set to non-trainable, for the second - they were updated during training The CNN´s architecture consists of Embedding layer with pretrained ```GloVe ``` weights, Convolutional Layer (CONV) with ReLU activation function, Global Max Pooling layer (GlobalMAXPOOL) and a Fully Connected Layer (FC). The output layer (OUT) uses softmax activation function and has 4 possible classes. 

Model´s architecture: Embedding -> CONV+ReLU -> GlobalMAXPOOL -> FC+ReLU -> OUT(softmax)

CNNs are prone to overfitting, therefore I applied a weight regularization method to CONV and FC layers to minimize the overfitting. I have used L2 regularization to constrain how the model performs (l2 = L2(0.0001)).

More about GloVe algorithm : https://nlp.stanford.edu/projects/glove/

Depiction of model´s architecture can be found in folder called ***'out'***.


## Repository contents

| File | Description |
| --- | --- |
| output | Folder containing files produced by the script |
| output/Emotions_classifier_report.csv | Classification metrics of the model |
| output/Emotions_classifier_performance.png | Model´s performance graph |
| output/VGG-Face_CNN´s_architecture.png | Depiction of CNN model´s architecture used |
| src/ | Folder containing the script |
| src/emotions_CNN&LR.py | The script |
| README.md | Description of the assignment and the instructions |
| emotion_venv.sh | bash file for creating a virtual environmment  |
| kill_emotion.sh | bash file for removing a virtual environment |
| requirements.txt| list of python packages required to run the script |



## Data

Dataset for this project consists of annotated tweets based on four emotions: joy, sadness, anger and fear. Manual annotation of the dataset to obtain real-valued scores was done through Best-Worst Scaling (BWS), an annotation scheme shown to obtain very reliable scores. I have merged the three datasets found in the link below, and split the data to training and testing sets in the script, to increase the amount of training data. Dataset has two columns: 'text', which is the tweet and 'label', which is an emotional category the tweet belongs to.

Link to data: https://www.kaggle.com/anjaneyatripathi/emotion-classification-nlp?select=emotion-labels-val.csv


___Data preprocessing___

The preprocessing of data for LR model included the following step:
- vecorizing training and test data using sklearn CountVectorizer(), which transformed text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.

The preprocessing of data for CNN model included the following steps:
- tokenizing training and test data using tensorflow.keras.Tokenizer() which quickly and efficiently convert text to numbers
- to make the Tokenizer output workable, the documents are padded to be of equal length (maxlen set according to a max line length)
- labels transformed to binarized vectors
- mapping the vocabulary in the data onto the pretrained embeddings, creating an embedding matrix with one row per word in the vocabulary

## Intructions to run the code

The code was tested on an HP computer with Windows 10 operating system. It was executed on Jupyter worker02.
Note:  ```Glove Word Embeddings``` are downloaded during the creation of virtual environment.

__Code parameters__


| Parameter | Description |
| --- | --- |
| data_dir  (dir) | Directory of the imput CSV file |
| test_size (test) | The size of the test data as a percentage, where the default = 0.25 (25%) |
| optimizer (optim) | Method to update the weight parameters to minimize the loss function. Default = Adam |
| epochs (ep) | Defines how many times the learning algorithm will work through the entire training dataset. Default = 60 |





__Steps__

Set-up:

```
#1 Open terminal on worker02 or locally
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/cds-language_Self_assigned_project.git  

#4 Navigate to the newly cloned repo
$ cd cds-language_Self_assigned_project

#5 Create virtual environment with its dependencies and activate it
$ bash create_emotion_venv.sh
$ source ./emotion/bin/activate

``` 

Run the code:

```
#6 Navigate to the directory of the script
$ cd src

#7 Run the code with default parameters
$ python emotions_CNN&LR.py -dir ../data/tweets_class.csv

#8 Run the code with self-chosen parameters
$ python emotions_CNN&LR.py -dir ../data/tweets_class.csv -test 0.3 -optim SGD -ep 100

#9 To remove the newly created virtual environment
$ bash kill_emotion_venv.sh

#10 To find out possible optional arguments for the script
$ python emotions_CNN.py --help


 ```

I hope it worked!


## Results

LR classifier achieved a weighted average accuracy of 86% for correctly classifying tweets based on emotion. CNN classifier achieved a weighted average accuracy of 77% using  pre-trained ```Glove Word Embeddings```, which were not updated during this training. The same CNN classifier achieved a weighted average accuracy of 84% using  pre-trained ```Glove Word Embeddings``` and updating them during the training. All of this indicate, that a simple LR model performs better than a deep learning CNN.

Performance graphs reveal, that the validation curve in both CNN model´s runs (with non-trainable and trainable weights) was decreasing for the first few epochs, then started increasing and fluctuating until the end of training, which might indicate an overfitting issue. Although both runs faced the same problem, overfitting was worse for the run with non-trainable weights, also, the validation accuracy was lower for this run. ```Glove Word Embeddings```  were trained on a combination of the Wikipedia 2014 dump and the Gigaword 5 corpus, which might not be very closely related to the language used on Twitter social media, which can be full of jargon. More experimenting is needed to draw final conclusions.


## References

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation





