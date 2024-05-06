# NLP-Course
Natural Language Processing Course in HUJi. From Basic to advance models. Solving: Classification, Word Embedding, Sentiment Analysis, Information Extraction, Semantic Role Labeling, Machine Translation, Pretraining and Transformers.


### Text Generation - Markov Languages Models
Unigram and Bigram models, Back off model, linear_interpolation_smoothing.


### Part-of-Speech Tagging - Hidden Markov Models
bigram HMM tagger, Viterbi algorithm, Add-one smoothing.


### Sentiment Analysis - Log Linear Model, Word2Vec representation, LSTM model
Build 3 models - Simple Log Linear, Log Linear with Word2Vec and LSTM.
On the Dataset of Sentiment Treebank by Stanford (movie reviews, and their sentiment value).

Using Pytorch.

Simple log-linear model:
This model will use a simple one-hot embedding for the words in order to perform the task.

Word2Vec log-linear model:
This model is almost identical to the simple log-linear model, except it uses pre-trained
Word2Vec embeddings instead of a simple one-hot embedding.

LSTM model:
In this model, each LSTM cell receives as input the Word2Vec embedding of a word in the
input sentence. Then the model takes the two hidden states of the LSTM layer (the last hidden
state of both directions of the bi-LSTM layer) – and concatenate them. Later, he put this
concatenation through a linear layer and finally output the sigmoid of the result.



