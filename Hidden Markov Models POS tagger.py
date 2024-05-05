import re
import nltk
from collections import defaultdict
from nltk.corpus import brown
brown.tagged_sents(categories='news')


def prepare_dataset():
    corpus_sentences = [item for item in brown.tagged_sents(categories='news')]
    for i in range(len(corpus_sentences)):
        #     corpus_sentences[i].insert(0, ('START', 'START'))
        #     corpus_sentences[i].append(('STOP', 'STOP'))
        for j in range(len(corpus_sentences[i])):
            word, tag = corpus_sentences[i][j]
            corpus_sentences[i][j] = (word, (re.split('[-+]', tag))[0])
    train = corpus_sentences[:int(len(corpus_sentences) * 0.9)]
    test = corpus_sentences[int(len(corpus_sentences) * 0.9):]
    return train, test


def get_tag_dict(train):
    big_dict = dict()
    for sent in train:
        for word, tag in sent:
            if word not in big_dict:
                big_dict[word] = defaultdict(int)
            big_dict[word][tag] += 1
    return big_dict


def find_most_likely_tag_baseline(train):
    print("==========================")
    print("Starting Question b:")
    ## Best tag for each word
    big_dict = get_tag_dict(train)
    max_like_dict = dict()
    for word in big_dict:
        max_like_dict[word] = max(big_dict[word], key=big_dict[word].get)
    return max_like_dict, big_dict


def find_error_rate(test, max_like_dict):
    ## Error rate
    known = correct_known = unknown = correct_unknown = 0
    for sent in test:
        for word, tag in sent:
            if word in tag_dict:
                known += 1
                if tag == max_like_dict[word]: correct_known += 1
            else:
                unknown += 1
                if tag == 'NN': correct_unknown += 1

    print("Result for b:")
    print('Known words error rate is {}'.format((1 - correct_known / known)))
    print('Unknown words error rate is {}'.format((1 - correct_unknown / unknown)))
    print('Total error rate is {}'.format((1 - (correct_known + correct_unknown) / (known + unknown))))
    print("Ending Question b.")


def compute_trans_prob(train):
    print("==========================")
    print("Starting Question c:")
    y_dict = dict()
    trans_prob = dict()
    for sent in train:
        sent = [('START', 'START')] + sent + [('STOP', 'STOP')]
        for i in range(1, len(sent)):
            yi_1 = sent[i - 1][1]
            yi = sent[i][1]
            if yi_1 not in y_dict:
                y_dict[yi_1] = defaultdict(float)
            y_dict[yi_1][yi] += 1
    for yi_1 in y_dict:
        trans_prob[yi_1] = defaultdict(float)
        denominator = sum(y_dict[yi_1].values())
        for yi in y_dict[yi_1]:
            trans_prob[yi_1][yi] = y_dict[yi_1][yi] / denominator
    return trans_prob


def compute_emission_probability(train):
    tags_dict = dict()
    emis_prob = dict()
    for sent in train:
        for word, tag in sent:
            if tag not in tags_dict:
                tags_dict[tag] = defaultdict(float)
            tags_dict[tag][word] += 1
    for tag in tags_dict:
        emis_prob[tag] = defaultdict(float)
        denominator = sum(tags_dict[tag].values())
        for word in tags_dict[tag]:
            emis_prob[tag][word] = tags_dict[tag][word] / denominator
    return tags_dict, emis_prob


def viterbi(sent, tags_dict, trans_prob, emis_prob, big_dict):
    n = len(sent)
    S = list()
    for i in range(n):  # Building table S
        S.append([[tag, 0, 0] for tag in tags_dict.keys()])
    for node in S[0]:  # First column
        e = emis_prob[node[0]][sent[0]]
        if sent[0] not in big_dict:
            e = 1
        node[1] = 'START'
        q = trans_prob['START'][node[0]]
        node[2] = q * e
    for j in range(1, n):  # Words
        for node in S[j]:  # Tags
            emis_prob_for_word = emis_prob[node[0]][sent[j]]
            if sent[j] not in big_dict:
                emis_prob_for_word = 1
            best_tag = S[j - 1][0]
            best_prob = S[j - 1][0][2] * trans_prob[S[j - 1][0][0]][node[0]] * emis_prob_for_word
            for last_tag in S[j - 1]:  # (cur_tag, pointer_to_previous_node, road_probability)
                cur_prob = last_tag[2] * trans_prob[last_tag[0]][node[0]] * emis_prob_for_word
                if cur_prob > best_prob:
                    best_prob, best_tag = cur_prob, last_tag
            node[1], node[2] = best_tag, best_prob
    best_arr = S[-1][0]
    for val in S[-1]:
        if val[2] > best_arr[2]:
            best_arr = val

    tags_list = [best_arr[0]]
    cur_tag = best_arr[1]
    while isinstance(cur_tag, list):
        tags_list.append(cur_tag[0])
        cur_tag = cur_tag[1]
    tags_list.reverse()
    return tags_list


def predict_vitrebi(test, tags_dict, emis_prob, trans_prob, big_dict):
    tags_res = []
    true_tags = []
    true_tags_list = []
    known = correct_known = unknown = correct_unknown = 0
    for sent in test:
        tags_list = viterbi(sent=[s[0] for s in sent], tags_dict=tags_dict, emis_prob=emis_prob, trans_prob=trans_prob,
                            big_dict=big_dict)
        true_tags_list = [word[1] for word in sent]
        assert(len(tags_list) == len(true_tags_list))
        for i in range(len(sent)):
            word, tag = sent[i]
            pred_tag = tags_list[i]
            if word in big_dict:
                known += 1
                if tag == pred_tag:
                    correct_known += 1
            else:
                unknown += 1
                if pred_tag == tag:
                    correct_unknown += 1
        tags_res.extend(tags_list)
        true_tags.extend(true_tags_list)
    print('Known words error rate is {}'.format((1 - correct_known / known)))
    print('Unknown words error rate is {}'.format((1 - correct_unknown / unknown)))
    print('Total error rate is {}'.format((1 - (correct_known + correct_unknown) / (known + unknown))))
    assert(len(tags_res) == len(true_tags))
    return tags_res, true_tags


def compute_emission_add_one(train, big_dict):
    print("==========================")
    print("Starting Question d:")
    tags_dict = dict()
    emis_prob = dict()
    for sent in train:
        for word, tag in sent:
            if tag not in tags_dict:
                tags_dict[tag] = defaultdict(int)
            tags_dict[tag][word] += 1
    for tag in tags_dict:
        emis_prob[tag] = defaultdict(float)
        for word in big_dict:
            denominator = sum(tags_dict[tag].values())
            emis_prob[tag][word] = (tags_dict[tag][word] + 1) / (denominator + len(big_dict))
    return tags_dict, emis_prob


# Emission Probabilities Using (Laplace) Add-one smoothing
def make_pseudo(word: str):
    print("==========================")
    print("Starting Question (e)(ii):")
    signs = [".", "$", ":", "/", "%", "-", "^", "!", "@", "#", "&", "*", "(", ")", "=", "+", r"\\", "`", ",", "<",
             ">", "\"", "[", "]", "{", "}", "|", ";", ";", "~"]
    categories = ["_Title", "_Number", "_Noun", "_Year", "_Range", "_Phrase", "_Belong", "_Time", "_Date", "_Ordinal", "_Age"
                  , "_Word", "_Null"]
    categories.extend(signs)
    if word in categories:
        return word
    if re.match("\d{4}(-\d{2})?", word):
        return "_Year"
    if re.match("\d+-\d+", word):
        return "_Range"
    if re.match(".+(-.+)+", word):
        return "_Phrase"
    if re.match("[A-Za-z]+'s", word):
        return "_Belong"
    if re.match("[0-9]+(:[0-9]+)+", word):
        return "_Time"
    if re.match("[0-9]+\.[0-9]+\.[0-9]+", word):
        return "_Date"
    if re.match("\d+(th|st|rd|nd)", word):
        return "_Ordinal"
    if word.isalpha() and word[0].isupper():
        return "_Title"
    if word.isnumeric():
        if len(word) == 2:
            return "_Age"
        return "_Number"
    for sign in signs:
        if sign in word:
            return "_" + sign
    if word.isalnum():
        return "_Word"
    return "_Null"


def prepare_pseudo_dataset(tag_dict):
    # Using Pseudo-Words
    unfreq_words = []
    for word in tag_dict:
        if sum(tag_dict[word].values()) < 5:
            unfreq_words.append(word)
    for i in range(len(train)):  # sentences
        for j in range(len(train[i])):  # words tuple: (word, tag)
            if (train[i][j][0] in unfreq_words):  # word in sentence in unfreq_words
                train[i][j] = (make_pseudo(train[i][j][0]), train[i][j][1])

    for i in range(len(test)):
        for j in range(len(test[i])):
            if (test[i][j][0] in unfreq_words):
                test[i][j] = (make_pseudo(test[i][j][0]), test[i][j][1])


def confusion_matrix(y_pred, y_true):
    # Confusion matrix
    tags = list(set(y_true))
    conf_mat = [[0 for t in tags] for tag in tags]
    for i in range(len(y_true)):
        conf_mat[tags.index(y_pred[i])][tags.index(y_true[i])] += 1
    # Investigate most frequent errors
    print("Investigate most frequent errors")
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat)):
            if i != j and conf_mat[i][j] > 35:
                print("True tag is {}, Predicted tag is {}, got wrong predict for {} times".format(tags[i], tags[j], conf_mat[i][j]))


if __name__ == "__main__":
    # 1) prepare NLTK corpus
    train, test = prepare_dataset()
    # 2) implementation of the most likely tag baseline - compute maximum likelyhood and error rate
    max_like_dict, tag_dict = find_most_likely_tag_baseline(train)
    find_error_rate(test, max_like_dict)
    # 3) Implementation of a bigram HMM tagger (transition and emission)
    trans_prob = compute_trans_prob(train)
    tags_dict, emis_prob = compute_emission_probability(train)
    # 4) Implement the Viterbi algorithm corresponding to the bigram HMM model.
    predict_vitrebi(test, tags_dict, emis_prob, trans_prob, tag_dict)
    # 5) Compute the emission probabilities of a bigram HMM tagger directly using (Laplace) Add-one smoothing
    tags_dict, emis_prob = compute_emission_add_one(train, tag_dict)
    predict_vitrebi(test, tags_dict, emis_prob, trans_prob, tag_dict)
    # 6) Design a set of pseudo-words for unknown words - try with viterbi and with Add-one smoothing.
    tag_dict = get_tag_dict(train)
    tags_dict, emis_prob = compute_emission_probability(train)
    trans_prob = compute_trans_prob(train)
    predict_vitrebi(test, tags_dict, emis_prob, trans_prob, tag_dict)
    # Using the pseudo-words AND add one smoothing
    tags_dict, emis_prob = compute_emission_add_one(train, tag_dict)
    trans_prob = compute_trans_prob(train)
    confusion_matrix(predict_vitrebi(test, tags_dict, emis_prob, trans_prob, tag_dict))
