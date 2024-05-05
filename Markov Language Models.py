import spacy
import numpy as np
from math import inf
from datasets import load_dataset
from collections import defaultdict


nlp = spacy.load("en_core_web_sm")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")


class UnigramModel:
    def __init__(self):
        self.reps_dict = defaultdict(int)
        self.probs_dict = dict()

    def fit(self, doc):
        for token in doc:
            if token.is_alpha:
                self.reps_dict[token.lemma_] += 1

    def make_probs(self):
        N = sum(self.reps_dict.values())
        for word, reps in self.reps_dict.items():
            self.probs_dict[word] = reps / N

    def predict_next_word(self, sent):
        return max(self.probs_dict, key=self.probs_dict.get)

    def predict_sent(self, sent):
        prob = 1
        for word in sent:
            if word.lemma_ not in self.probs_dict.keys():
                return 0
            prob *= self.probs_dict[word.lemma_]

        return prob

class BigramModel:
    def __init__(self):
        self.reps_dict = defaultdict(int)
        self.bi_dict = defaultdict(int)
        self.probs_dict = dict()

    def fit(self, doc):
        prev_token = 'START'
        for token in doc:
            if token.is_alpha:
                self.bi_dict[(prev_token, token.lemma_)] += 1
                self.reps_dict[prev_token] += 1
                prev_token = token.lemma_

    def make_probs(self):
        for bi, reps in self.bi_dict.items():
            self.probs_dict[bi] = reps / self.reps_dict[bi[0]]

    def predict_next_word(self, sent):
        if len(self.probs_dict) == 0:
            raise Exception(f"No train has been done. self.probs_dict: {self.probs_dict}")

        pred_word = ""
        best_prob = -inf
        last_word = sent[-1].lemma_
        for bi, prob in self.probs_dict.items():
            if bi[0] == last_word and prob > best_prob:
                best_prob = prob
                pred_word = bi[1]
        return pred_word

    def predict_sent(self, sent):
        if ('START', sent[0].lemma_) not in self.probs_dict:
            return 0


        prob = self.probs_dict[('START', sent[0].lemma_)]
        for i in range(1, len(sent)):
            if (sent[i - 1].lemma_, sent[i].lemma_) not in self.probs_dict:
                return 0
            prob *= self.probs_dict[(sent[i - 1].lemma_, sent[i].lemma_)]

        return prob

    def perplexity(self, lst):
        M, probs = 0, 0
        for sent in lst:
            M += len(sent)
            temp_prob = self.predict_sent(sent)
            if temp_prob == 0:
                return 0
            probs *= temp_prob
        l = probs / M
        return 2 ** (-l)


# Task 1 - Train the models
def train_models():
    uni_model = UnigramModel()
    bi_model = BigramModel()

    for text in dataset['text']:
        doc = nlp(text)
        uni_model.fit(doc)
        bi_model.fit(doc)

    uni_model.make_probs()
    bi_model.make_probs()

    return uni_model, bi_model


# Task 2 - find the most probable word predicted by the model: “ I have a house in ...”.
def use_model(bi_model):
    SENT = nlp('I have a house in')
    print("Task 2 answer is: ", bi_model.predict_next_word(SENT))


# Task 3 - compute probability and perplexity
def compute_probability_and_perplexity(bi_model):
    SENT1 = nlp('Brad Pitt was born in Oklahoma')
    SENT2 = nlp('The actor was born in USA')

    prob_sent1 = bi_model.predict_sent(SENT1)
    prob_sent2 = bi_model.predict_sent(SENT2)
    print("Task 3A.1 answer is: ", np.log(prob_sent1) if prob_sent1 != 0 else "Probability is 0. Operation log does not work on value zero")
    print("Task 3A.2 answer is: ", np.log(prob_sent2) if prob_sent2 != 0 else "Probability is 0. Operation log does not work on value zero")

    arr = [SENT1, SENT2]
    perplexity = bi_model.perplexity(arr)
    print("Task 3B answer is: ", np.log(perplexity) if perplexity != 0 else "Undefined. probability is 0. Most likely because combination not in corpus")


# Task 4 - linear interpolation smoothing between the two models
class BackOffModel:
    def __init__(self, l1, l2, uni=UnigramModel(), bi=BigramModel()):
        self.l1 = l1
        self.l2 = l2
        self.uni = uni
        self.bi = bi

    def predict_sent(self, sent):
        # contain probabilities
        words = []
        # first word
        l2_prob = 1
        l1_prob = 0 if sent[0].lemma_ not in self.uni.probs_dict.keys() else self.uni.probs_dict[sent[0].lemma_]
        if ('START', sent[0].lemma_) not in self.bi.probs_dict:
            words.append(l1_prob*self.l1 + 0)
        else:
            l2_prob = self.bi.probs_dict[('START', sent[0].lemma_)]
            words.append(l1_prob*self.l1 + l2_prob*self.l2)
        # rest of the words
        for i in range(1, len(sent)):
            if (sent[i - 1].lemma_, sent[i].lemma_) not in self.bi.probs_dict:
                l2_prob = 0
            else:
                l2_prob = self.bi.probs_dict[(sent[i - 1].lemma_, sent[i].lemma_)]
            l1_prob = 0 if sent[i].lemma_ not in self.uni.probs_dict.keys() else self.uni.probs_dict[sent[i].lemma_]
            words.append(l1_prob*self.l1 + l2_prob*self.l2)
        # calc final probability
        prob = 1
        for pr in words:
            prob *= pr
        return prob

    def perplexity(self, lst):
        M, probs = 0, 0
        for sent in lst:
            M += len(sent)
            temp_prob = np.log(self.predict_sent(sent))
            probs += temp_prob
        l = probs / M
        return np.e ** (-l)


def linear_interpolation_smoothing(uni_model, bi_model):
    SENT1 = nlp('Brad Pitt was born in Oklahoma')
    SENT2 = nlp('The actor was born in USA')
    l_bigram = 2 / 3
    l_unigram = 1 / 3
    back_off_model = BackOffModel(l_unigram, l_bigram, uni_model, bi_model)

    # Probabilities
    prob_sent1 = back_off_model.predict_sent(SENT1)
    prob_sent2 = back_off_model.predict_sent(SENT2)
    print("Task 4A.1 answer is: ", np.log(prob_sent1))
    print("Task 4A.2 answer is: ", np.log(prob_sent2))

    # Perplexity
    arr = [SENT1, SENT2]
    print("Task 4B answer is: ", back_off_model.perplexity(arr))


if __name__ == "__main__":
    uni_model, bi_model = train_models()
    use_model(bi_model)
    compute_probability_and_perplexity(bi_model)
    linear_interpolation_smoothing(uni_model, bi_model)