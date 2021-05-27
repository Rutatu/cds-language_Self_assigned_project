#!/usr/bin/env python



''' ---------------- About the script ----------------

Self-assigned project: Text Classification Using CNN and Pre-trained Glove Word Embeddings: tweets classification based on emotions


This script builds a Logistic Regression (LR) model and a deep learning model using convolutional neural networks (DL CNN) which classify tweets based on 4 emotions: joy, sadness, anger and fear. DL CNN model uses pre-trained Glove Word Embeddings

The script trains a Logistic Regression (LR) model on tweets to establish a baseline model performance, it outputs a classification report and a confusion matrix, prints out results to the terminal. Then, it trains a deep learning CNN model using pre-trained Glove Word Embeddings and outputs classification report anf performance graph, prints out performance results to the terminal.


Arguments:
    
    -dir,    --data_dir:           Directory of the CSV file
    -test,   --test_size:          The size of the test data as a percentage, where the default = 0.25 (25%)
    -optim,  --optimizer:          Method to update the weight parameters to minimize the loss function. Default = Adam
    -ep,     --epochs:             Defines how many times the learning algorithm will work through the entire training dataset. Default = 60




Example:    
    
    with default arguments:
        $ python emotions_CNN&LR.py -dir ../data/tweets_class.csv
        
    with optional arguments:
        $ python emotions_CNN&LR.py -dir ../data/tweets_class.csv -test 0.3 -optim SGD -ep 50


'''




"""---------------- Importing libraries ----------------
"""


# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf_util

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D, Dropout, LSTM)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import plot_model


#from tensorflow.keras.constraints import maxnorm

# matplotlib
import matplotlib.pyplot as plt

# Command-line interface
import argparse



"""---------------- Main script ----------------
"""


def main():
    
    """------ Argparse parameters ------
    """
    # Instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser(description = "[INFO] Classify tweets based on four emotions:anger, fear, joy, sadness, and print out performance accuracy report")
    
    # Adding arguments
    parser.add_argument("-dir", "--data_dir", required = True, help = "Directory of the CSV file")
    parser.add_argument("-test", "--test_size", required=False, default = 0.25, type = float, help = "The size of the test data as a percentage, where the default = 0.25 (25%)")
    parser.add_argument("-optim", "--optimizer", required = False, default = 'adam', help = "Method to update the weight parameters to minimize the loss function. Default = Adam")
    parser.add_argument("-ep", "--epochs", required=False, default = 60, type = int, help = "Defines how many times the learning algorithm will work through the entire training dataset. Default = 60")
    
                                          
    # Parsing the arguments
    args = vars(parser.parse_args())
    
    # Saving parameters as variables
    data = args["data_dir"] # Directory of the CSV file
    test = args["test_size"] # The size of the test data set
    optim = args["optimizer"] # Optimizer
    ep = args["epochs"] # epochs
     
     
    

    """------ Loading data and preprocessing ------
    """

    # Message to a user
    print("\n[INFO] Loading data and preparing for training a Deep Learning model...")
    
    # Create ouput folder, if it doesn´t exist already, for saving the classification report, performance graph and model´s architecture 
    if not os.path.exists("../out"):
        os.makedirs("../out")
    
    # Loading and reading data
    filename = os.path.join(data)
    data = pd.read_csv(filename)
        
    # Extracting sentences and seasons for creating training and test data sets
    tweets = data['text'].values
    labels = data['label'].values

    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(tweets, 
                                                        labels, 
                                                        test_size=test, 
                                                        random_state=42)

  
    """------ Logistic Regression model ------
    """

    # Initializing vectorizer
    vectorizer = CountVectorizer()

    # First I do it for our training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then I do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    
    print("[INFO] training Logistic Regression model...")
    classifier = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_feats, y_train)

    print("[INFO] evaluating Logistic Regression model...")
    # Predicted labels
    y_pred = classifier.predict(X_test_feats)
     
    # CLASSIFIER REPORT
    # Logistic Regression classifier report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)  
    # Turning classification report into a dataframe
    report_df = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True)).transpose()    
    # Defining full filepath to save csv file 
    outfile = os.path.join("..", "out", "logReg_classification_report.csv")
    # Saving a dataframe as .csv
    report_df.to_csv(outfile)  
    # Printing that .csv file has been saved
    print(f"\n[INFO] classification report is saved in directory {outfile}")
        
     
    # CONFUSION MATRIX
    # Defining full filepath to save .png file
    path_png = os.path.join("..", "out", "logReg_confusion_matrix.png")
    # Creating confusion matrix
    clf_util.plot_cm(y_test, y_pred, normalized=True)
    # Saving as .png file
    plt.savefig(path_png)
    # Printing that .png file has been saved
    print(f"\n[INFO] confusion matrix is saved in directory {path_png}")
    
    
    
    
   
    """------ CNN model: preprocessing ------
    """
   
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=5000)
    # Fit to training data
    tokenizer.fit_on_texts(X_train)

    # Tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # Overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    
    # Finding max length of lines to use for finding padding maxlen
    # Empty list
    k = []
    # For loop to append length of each line in the training data
    for i in X_train:
        # append to empty list
        k.append(len(i))
    # Define maxlen
    maxlen = max(k) + 4
    
    
    
    # Pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding='post', # sequences can be padded "pre" or "post"
                                maxlen=maxlen)
    # Pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                               padding='post', 
                               maxlen=maxlen)

    # Transform labels to binarized vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(y_train)
    testY = lb.fit_transform(y_test)


        
    """------ Defining CNN model ------
    """
    
    tf.keras.backend.clear_session()
    # Define regularizer
    l2 = L2(0.0001)
        
    # Define embedding size we want to work with
    embedding_dim = 50

    # Create embedding matrix
    embedding_matrix = create_embedding_matrix('../data/glove/glove.6B.100d.txt',
                                           tokenizer.word_index, 
                                           embedding_dim)
    
    # Define model
    model = Sequential()

    # Embedding -> CONV+ReLU -> MaxPool -> FC+ReLU -> Out
    model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                        embedding_dim,               # embedding input layer size
                        weights=[embedding_matrix],  # pretrained GloVe weights
                        input_length=maxlen,         # maxlen of padded doc      
                        trainable=False))             # trainable embeddings                 
                                           
 

    model.add(Conv1D(256, 5, 
                    activation='relu',
                    kernel_regularizer=l2))          # L2 regularization 
    model.add(GlobalMaxPool1D())
    
        
    model.add(Dense(128, activation='relu', kernel_regularizer=l2))


    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', # adding 'sparse_' solved error with incompatible shape using only 'categorical_crossentropy' 
                  optimizer=optim,
                  metrics=['accuracy'])

    # Print summary
    model.summary()
    
    # Ploting and saving model´s architecture
    plot_model(model, to_file='../out/Model´s_architecture.png',
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    
    # Printing that model´s architecture graph has been saved
    print(f"\n[INFO] Deep learning model´s architecture graph has been saved")
    
    
            
    """------ Training and evaluating CNN model ------
    """
    
    print("[INFO] training and evaluating Deep Learning model ...")
    history = model.fit(X_train_pad, trainY,
                    epochs=ep,
                    verbose=True,
                    validation_data=(X_test_pad, testY))
                 

    # Evaluate 
    loss, accuracy = model.evaluate(X_train_pad, trainY, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, testY, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # Plot
    plot_history(history, epochs = ep) 

    # Printing that performance graph has been saved
    print(f"\n[INFO] Deep Learning CNN´s performance graph has been saved")
    
    # Labels
    labels = ["anger", "fear", "joy", "sadness"]
    
    # Classification report
    predictions = model.predict(X_test_pad, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels))
    
    
    
    # Defining full filepath to save .csv file 
    outfile = os.path.join("../", "out", "Emotions_classifier_report.csv")
    
    # Turning report into dataframe and saving as .csv
    report = pd.DataFrame(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labels, output_dict = True)).transpose()
    report.to_csv(outfile)
    print(f"\n[INFO] Emotions classification report has been saved")

    
    print("\nScript was executed successfully! Have a nice day")
        
    

"""---------------- Functions ----------------
"""

# These functions were developed for use in class and have been adapted for this project

def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig('../out/Emotions_classification_performance_graph.png')
    

    
    
    
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
    
    
    
    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()

