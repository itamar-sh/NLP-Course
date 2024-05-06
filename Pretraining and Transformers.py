###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################

import numpy as np
"""
requirements:
pip install git+https://github.com/huggingface/transformers
pip install -U scikit-learn
pip install datasets
"""

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    tf_train = tf.fit_transform(x_train)
    clf = LogisticRegression(random_state=0).fit(tf_train, y_train)
    tf_test = tf.transform(x_test)
    return clf.score(tf_test, y_test)


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")
    training_args = TrainingArguments(output_dir='./fine_tuning',
                                      learning_rate=5e-5,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      num_train_epochs=5)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    train_data = Dataset(tokenizer(x_train, max_length=500), y_train)
    test_data = Dataset(tokenizer(x_test, max_length=500), y_test)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      eval_dataset=test_data,
                      tokenizer=tokenizer)
    trainer.train()
    predictions = trainer.predict(test_data)
    return compute_metrics((predictions.predictions, y_test))


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
    candidate_labels = list(category_dict.values())
    predictions = clf(x_test, candidate_labels)
    reverse_category_dict = {'computer graphics': 0, 'baseball': 1, 'science, electronics': 2, 'politics, guns': 3}
    y_pred = [reverse_category_dict[predictions[i]['labels'][0]] for i in range(len(predictions))]
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == "__main__":
    dataset_portions = [0.1, 0.5, 1.]
    # Q1
    print("Logistic regression results:")
    for portion in dataset_portions:
        print(f"Portion: {portion}")
        print(linear_classification(portion))

    # Q2
    print("Finetuning results:")
    for portion in dataset_portions:
        print(f"Portion: {portion}")
        print(transformer_classification(portion))

    # Q3
    print("Zero-shot result:")
    print(zeroshot_classification())
