### Named Entity Recognition using GMB Dataset and Spacy Library

This repository contains an implementation of Named Entity Recognition (NER), which is a type of text classification, using the GMB Dataset and the Spacy library. The goal of NER is to identify and classify entities such as people, organizations, locations, and dates in unstructured text.

#### Mathematical Foundation

The mathematical foundation of NER involves the use of statistical models such as Hidden Markov Models (HMMs), Conditional Random Fields (CRFs), and Deep Neural Networks (DNNs). In this implementation, a DNN model is used to classify entities based on their context in the text.

The DNN model consists of an input layer, one or more hidden layers, and an output layer. The input layer takes as input the word representation of the text, which can be obtained using techniques such as word embedding. The hidden layers are used to extract meaningful features from the input layer, and the output layer is used to make the final classification decision.

The loss function used in the DNN model is the cross-entropy loss function, which measures the difference between the predicted probabilities and the true labels of the entities. The cross-entropy loss function is defined as:

$$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} y_{ij} \log \hat{y}_{ij} $$

where `N` is the number of samples, `K` is the number of classes, `y` is the one-hot encoded true label of the sample, and $\hat{y}$ is the predicted probability of the sample belonging to each class.

The goal of training the DNN model is to find the weights that minimize the cross-entropy loss function. This can be achieved through an optimization algorithm such as gradient descent.

#### Data Preprocessing and Model Training

The GMB Dataset is preprocessed to extract the text and entities, and the text is converted into word representations using word embedding. The DNN model is then trained on the preprocessed data, and the optimization algorithm is used to minimize the cross-entropy loss function.

#### Results and Evaluations

The trained DNN model is evaluated on a separate test dataset, and the results are compared with the true entities to assess the accuracy of the model. One measure of accuracy is the F1 score, which is the harmonic mean of precision and recall. A high-quality model will have a high F1 score, indicating that it is able to accurately identify entities while avoiding false positive and false negative errors.

In this example, the confusion matrix indicates that the model has a high degree of accuracy, with high values along the main diagonal and low values in the off-diagonal entries. This indicates that the model is able to accurately identify the entities in the text.

#### Usage

The code for the Named Entity Recognition model can be found in the `NER - Spacy.ipynb` file in this repository. The dataset for the same can be found in the `ner_dataset.txt` file. The GMB Dataset can be downloaded from the CoNLL 2002 website.

