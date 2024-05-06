import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import data_loader
import pickle
import matplotlib.pyplot as plt


# ------------------------------------------- Constants ----------------------------------------


SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
RARE_WORDS = 'rare words'
NEGATED_POLARITY = 'negated polarity'


# ------------------------------------------ Helper methods and classes --------------------------


def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api

    wv_from_bin = api.load("word2vec-google-news-300")

    # vocab = list(wv_from_bin.vocab.keys())
    # print(wv_from_bin.vocab[vocab[0]])
    # print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim=300):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    sent, vec, total = sent.text, np.zeros(embedding_dim).astype('float32'), 0
    for word in sent:
        if word in word_to_vec:  # Known word
            vec += word_to_vec[word]
            total += 1
    return vec / total if total != 0 else vec


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size)
    vec[ind] += 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return: ndarray represents the averaged one-hot vector
    """
    vec_size = len(word_to_ind)
    vec = np.zeros(vec_size).astype('float32')
    for word in sent.text:
        vec += get_one_hot(vec_size, word_to_ind[word])
    return vec / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    indices_dict, cur_idx = dict(), 0
    for i in range(len(words_list)):
        indices_dict[words_list[i]] = words_list.index(words_list[i])
    return indices_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    word_embedding_lst = []
    for word in sent.text:
        word_embedding_lst.append(word_to_vec.get(word, np.zeros(embedding_dim)))
    if len(word_embedding_lst) >= seq_len:
        return np.array(word_embedding_lst)[:seq_len]
    return np.concatenate((np.array(word_embedding_lst), np.zeros((seq_len-len(word_embedding_lst), embedding_dim))))


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path,
                                                               split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[
                TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # Self Added - Special Subsets
        np_indices = data_loader.get_negated_polarity_examples(
            self.sentences[TEST])
        self.sentences[NEGATED_POLARITY] = np.take(self.sentences[TEST],
                                                   np_indices)

        rare_indices = data_loader.get_rare_words_examples(
            self.sentences[TEST], self.sentiment_dataset)
        self.sentences[RARE_WORDS] = np.take(self.sentences[TEST],
                                             rare_indices)

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(
                                         words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {
                "word_to_vec": create_or_load_slim_w2v(words_list),
                "embedding_dim": embedding_dim
                }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {
            k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
            for
            k, sentences in self.sentences.items()}
        self.torch_iterators = {
            k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
            for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN, VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array(
            [sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------


class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    # embedding_dim = 300, hidden_dim = 100, n_layers = 2
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.biLSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                              bidirectional=True, batch_first=True)
        self.hidden_dimensions = hidden_dim
        self.linear = nn.Linear(2 * hidden_dim, 1)
        self.layers = n_layers
        self.drop = nn.Dropout(dropout)
        self.name = 'LSTM'

    def forward(self, text):
        # text: shape=(64, 300)
        text2 = text.float()
        out, hidden = self.biLSTM(text2)
        cat = torch.cat((hidden[0][0], hidden[0][1]), 1)
        val = self.drop(cat)
        return self.linear(val)

    def predict(self, text):
        return nn.Sigmoid()(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.name = "Log Linear"

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return self.sigmoid(self.forward(x))


# ------------------------- Training Functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    accurate, vals = 0, len(preds)
    for i in range(vals):
        if preds[i] >= 0.5:
            cur_pred = 1
        else:
            cur_pred = 0
        if cur_pred == y[i]:
            accurate += 1
    return accurate / vals


def train_epoch(model, data_iterator, optimizer, criterion, epoch):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    accuracy, loss, iterations = 0, 0, 0
    num_full_batches = int(len(data_iterator.dataset) /
                           data_iterator.batch_size)
    limit = num_full_batches * data_iterator.batch_size
    for x_i, y_i in data_iterator:
        optimizer.zero_grad()
        preds = model(x_i)
        accuracy += binary_accuracy(model.predict(x_i.type(torch.FloatTensor)), y_i)
        cur_loss = criterion(preds.double(), torch.reshape(y_i, (len(y_i), 1)))
        cur_loss.backward()
        loss += cur_loss.item()
        optimizer.step()
        iterations += 1
        # Show Progress
        if iterations % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: '
                '{}/{} ({:.0f}%)'.format(
                    epoch,
                    iterations * len(x_i), limit, 100. * iterations /
                    num_full_batches, cur_loss.item(),
                    accuracy, (iterations + 1),
                    100. * accuracy / ((iterations + 1))))
    return loss / iterations, accuracy / iterations


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    with torch.no_grad():
        loss = accuracy = iterations = 0
        for x_i, y_i in data_iterator:
            preds = model.forward(x_i)
            loss += criterion(preds.double(), torch.reshape(y_i, (len(y_i), 1)))
            accuracy += binary_accuracy(preds, y_i)
            iterations += 1
    return loss / iterations, accuracy / iterations


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return: ndarray of all models predictions.
    """
    with torch.no_grad():
        preds = list()
        for x_i, y_i in data_iter:
            val = model.predict(x_i).flatten()
            val = val.tolist()
            preds.extend(val)
    return preds  # error because it's a list


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(model.parameters(), lr,
                                 weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(get_available_device())  # Loss Func

    # nn - Neural Network
    train_loss, train_accuracy, val_loss, val_accuracy =\
        np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)  # init

    for epoch in range(n_epochs):
        # optimizer: doing GD without step by step
        train_iterator = data_manager.get_torch_iterator(TRAIN)  # Get iterator on all the train
        train_loss[epoch], train_accuracy[epoch] = \
            train_epoch(model, train_iterator, optimizer, criterion, epoch)

        val_iterator = data_manager.get_torch_iterator(VAL)
        val_loss[epoch], val_accuracy[epoch] = \
            evaluate(model, val_iterator, criterion)

        PATH = "Train {}".format(model.name)
        save_model(model, PATH, epoch, optimizer)

        print('After Epoch {}, Train Loss = {}, Val Loss = {}'.format(epoch+1, train_loss[epoch], val_loss[epoch]))
        print('After Epoch {}, Train Acc = {}, Val Acc = {}'.format(epoch+1, train_accuracy[epoch], val_accuracy[epoch]))

    print('After All Epoches, By order Train Loss, Val Loss, Train Acc, Val Acc:')
    print(train_loss)
    print(val_loss)
    print(train_accuracy)
    print(val_accuracy)
    return train_loss, train_accuracy, val_loss, val_accuracy


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    LR = 0.01
    EPOCHS = 20
    DECAY = 0.001
    BATCH_SIZE = 64

    data_manager = DataManager(ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    embedding_dim = len(data_manager.sentiment_dataset.get_word_counts())  # size of one hot vector - 16721
    model = LogLinear(embedding_dim)

    train_loss, train_accuracy, val_loss, val_accuracy =\
        train_model(model, data_manager, EPOCHS, LR, DECAY)

    # Plot of train & validation loss values, as a function of the epoch number
    plt.figure(1)
    plt.plot(np.arange(1, EPOCHS+1), train_loss, color='b', label='Train Loss')
    plt.plot(np.arange(1, EPOCHS+1), val_loss, color='r', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "One Hot - Train & Validation Loss, As a Function of Epoch Number")
    plt.legend()
    # plt.savefig('One Hot - Train & Validation Loss, As a Function of Epoch Number')
    plt.show()

    # Plot of train & validation accuracy values, as a function of the epoch number
    plt.figure(2)
    plt.plot(np.arange(1, EPOCHS+1), train_accuracy, color='b',
             label='Train Accuracy')
    plt.plot(np.arange(1, EPOCHS+1), val_accuracy, color='r',
             label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "One Hot - Train & Validation Accuracy, As a Function of Epoch Number")
    plt.legend()
    # plt.savefig('One Hot - Train & Validation Accuracy, As a Function of Epoch Number')
    plt.show()

    # Compute Test Loss and Accuracy
    test_iter = data_manager.get_torch_iterator(TEST)
    test_preds = get_predictions_for_data(model, test_iter)
    y = data_manager.get_labels(TEST)
    test_accuracy = binary_accuracy(test_preds, y)
    criterion = nn.BCEWithLogitsLoss().to(get_available_device())  # Loss Func
    test_loss = criterion(torch.as_tensor(test_preds).double(), torch.from_numpy(y))

    print('Test Loss for Log Linear with One Hot is {}'.format(test_loss))
    print('Test Accuracy for Log Linear with One Hot is {}'.format(
        test_accuracy))

    # Negated Polarity
    np_iter = data_manager.get_torch_iterator(NEGATED_POLARITY)
    np_preds = get_predictions_for_data(model, np_iter)
    y = data_manager.get_labels(NEGATED_POLARITY)
    np_accuracy = binary_accuracy(np_preds, y)
    print(
        'The Accuracy Over Negated Polarity for Log Linear with One Hot is {}'.format(
            np_accuracy))

    # Rare Words
    rw_iter = data_manager.get_torch_iterator(RARE_WORDS)
    rw_preds = get_predictions_for_data(model, rw_iter)
    y = data_manager.get_labels(RARE_WORDS)
    np_accuracy = binary_accuracy(rw_preds, y)
    print(
        'The Accuracy Over Rare Words for Log Linear with One Hot is {}'.format(
            np_accuracy))


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    LR = 0.01
    EPOCHS = 20
    BATCH_SIZE = 64
    DECAY = 0.001

    data_manager = DataManager(W2V_AVERAGE, batch_size=BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(W2V_EMBEDDING_DIM).to(get_available_device())

    train_loss, train_accuracy, val_loss, val_accuracy =\
        train_model(model, data_manager, EPOCHS, LR, DECAY)

    # Plot of train & validation loss values, as a function of the epoch number
    plt.plot(np.arange(1, EPOCHS+1), train_loss, color='b', label='Train Loss')
    plt.plot(np.arange(1, EPOCHS+1), val_loss, color='r', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("W2V - Train & Validation Loss, As a Function of Epoch Number")
    plt.legend()
    # plt.savefig('W2V - Train & Validation Loss, As a Function of Epoch Number')
    plt.show()

    # Plot of train & validation accuracy values, as a function of the epoch number
    plt.plot(np.arange(1, EPOCHS+1), train_accuracy, color='b',
             label='Train Accuracy')
    plt.plot(np.arange(1, EPOCHS+1), val_accuracy, color='r',
             label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "W2V - Train & Validation Accuracy, As a Function of Epoch Number")
    plt.legend()
    # plt.savefig('W2V - Train & Validation Accuracy, As a Function of Epoch Number')
    plt.show()

    # Compute Test Loss and Accuracy
    test_iter = data_manager.get_torch_iterator(TEST)
    test_preds = get_predictions_for_data(model, test_iter)
    y = data_manager.get_labels(TEST)
    test_accuracy = binary_accuracy(test_preds, y)
    criterion = nn.BCEWithLogitsLoss().to(get_available_device())
    test_loss = criterion(torch.as_tensor(test_preds).double(), torch.from_numpy(y))

    print('Test Loss for Log Linear with W2V is {}'.format(test_loss))
    print('Test Accuracy for Log Linear with W2V is {}'.format(test_accuracy))

    # Negated Polarity
    np_iter = data_manager.get_torch_iterator(NEGATED_POLARITY)
    np_preds = get_predictions_for_data(model, np_iter)
    y = data_manager.get_labels(NEGATED_POLARITY)
    np_accuracy = binary_accuracy(np_preds, y)
    print(
        'The Accuracy Over Negated Polarity for Log Linear with W2V is {}'.format(
            np_accuracy))

    # Rare Words
    rw_iter = data_manager.get_torch_iterator(RARE_WORDS)
    rw_preds = get_predictions_for_data(model, rw_iter)
    y = data_manager.get_labels(RARE_WORDS)
    np_accuracy = binary_accuracy(rw_preds, y)
    print('The Accuracy Over Rare Words for Log Linear with W2V is {}'.format(
        np_accuracy))


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    LR = 0.001
    DECAY = 0.0001
    DROPOUT = 0.5
    BATCH_SIZE = 64
    EPOCHS = 4

    HIDDEN_DIMS = 100
    LAYERS = 2

    data_manager = DataManager(W2V_SEQUENCE, batch_size=BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(W2V_EMBEDDING_DIM, HIDDEN_DIMS, LAYERS, DROPOUT).to(
        get_available_device())

    train_loss, train_accuracy, val_loss, val_accuracy =\
        train_model(model, data_manager, EPOCHS, LR, DECAY)

    # Plot of train & validation loss values, as a function of the epoch number
    plt.plot(np.arange(EPOCHS), train_loss, color='b', label='Train Loss')
    plt.plot(np.arange(EPOCHS), val_loss, color='r', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Train & Validation Loss, As a Function of Epoch Number")
    plt.legend()
    # plt.savefig('LSTM - Train & Validation Loss, As a Function of Epoch Number')
    plt.show()

    # Plot of train & validation accuracy values, as a function of the epoch number
    plt.plot(np.arange(EPOCHS), train_accuracy, color='b',
             label='Train Accuracy')
    plt.plot(np.arange(EPOCHS), val_accuracy, color='r',
             label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "LSTM - Train & Validation Accuracy, As a Function of Epoch Number")
    plt.legend()
    # plt.savefig('LSTM - Train & Validation Accuracy, As a Function of Epoch Number')
    plt.show()

    # Compute Test Loss and Accuracy
    test_iter = data_manager.get_torch_iterator(TEST)
    test_preds = get_predictions_for_data(model, test_iter)
    y = data_manager.get_labels(TEST)
    test_accuracy = binary_accuracy(test_preds, y)
    criterion = nn.BCEWithLogitsLoss().to(get_available_device())
    test_loss = criterion(torch.as_tensor(test_preds).double(), torch.from_numpy(y)).to(get_available_device())

    print('Test Loss for LSTM is {}'.format(test_loss))
    print('Test Accuracy for LSTM is {}'.format(test_accuracy))

    # Negated Polarity
    np_iter = data_manager.get_torch_iterator(NEGATED_POLARITY)
    np_preds = get_predictions_for_data(model, np_iter)
    y = data_manager.get_labels(NEGATED_POLARITY)
    np_accuracy = binary_accuracy(np_preds, y)
    print('The Accuracy Over Negated Polarity for LSTM is {}'.format(
        np_accuracy))

    # Rare Words
    rw_iter = data_manager.get_torch_iterator(RARE_WORDS)
    rw_preds = get_predictions_for_data(model, rw_iter)
    y = data_manager.get_labels(RARE_WORDS)
    np_accuracy = binary_accuracy(rw_preds, y)
    print('The Accuracy Over Rare Words for LSTM is {}'.format(np_accuracy))


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()