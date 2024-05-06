import nltk
import random
import numpy as np
from networkx import DiGraph
from networkx.algorithms import minimum_spanning_arborescence, maximum_spanning_arborescence
from nltk.corpus import dependency_treebank


# COSTANTS
SPLIT = .9
LR = 1
ITERATIONS = 2
ADDR_IND = 0
WORD_IND = 1
TAG_IND = 2
HEAD_IND = 3


class Arc:
    """
    Arc object for using Chu Lui algorithm.
    """

    def __init__(self, head, tail, features, weight=.0):
        """
        :param head: Node of the head side of the arc.
        :param tail: Node of the tail side of the arc.
        :param features: List of 2 ints, each one is the feature index that
         should be set to '1' (features[0] = Word bigrams,
         features[1] = POS bigrams), as described in the PDF.
        :param weight: Float weight of the arc.
        """
        self.head = head
        self.tail = tail
        self.features = features
        self.weight = weight


class SentTree:
    """
    Tree object represents a sentence.
    """

    def __init__(self, nodes, data_container):
        self.data_container = data_container
        self.nodes = SentTree.fix_nodes(nodes)
        self.init_root(self.nodes)
        self.arcs = self.create_all_arcs()
        self.gold_tree = self.create_gold_tree()

    def create_all_arcs(self):
        """
        Create all arcs possible in the tree with their features.
        :return: List contains all Arc objects possible for this tree.
        """
        arcs_arr = list()
        for u in self.nodes:
            for v in self.nodes[1:]:  # Prevent root as tail
                if u == v: continue  # Prevent self arc
                features = self.data_container.feature_func(u, v)
                arcs_arr.append(Arc(u[ADDR_IND], v[ADDR_IND], features))
        return arcs_arr

    def create_gold_tree(self):
        """
        Creates the gold tree for the current sentence.
        :return: List contains all Arc objects of the gold tree.
        """
        # Create the arcs
        gold_arcs = list()
        for node in self.nodes[1:]:  # Skip root
            gold_arcs.append((self.nodes[node[HEAD_IND]], node))
        # Create arcs with features for gold tree edges
        arcs_arr = list()
        for (u, v) in gold_arcs:
            feature = self.data_container.feature_func(u, v)
            arcs_arr.append(Arc(u[ADDR_IND], v[ADDR_IND], feature))
        return arcs_arr

    # === STATIC METHODS ===
    @staticmethod
    def fix_nodes(nodes):
        """
        Keeps the needed data from each node of the parsed tree in a list.
        :param nodes: The nodes to take the data from.
        :return: List of nodes with the needed data.
        """
        nodes_arr = []
        for i in range(len(nodes)):
            # if nodes[i]['word'] not in [',', '.']:
            nodes_arr.append([nodes[i]['address'], nodes[i]['word'],
                              nodes[i]['tag'], nodes[i]['head']])
        return nodes_arr

    @staticmethod
    def init_root(nodes):
        """
        Initialize the root as said in the PDF.
        """
        nodes[0][WORD_IND] = 'ROOT'
        nodes[0][TAG_IND] = 'ROOT'


class DataContainer:
    """
    class Represents all trees for all sentences in the train data.
    """

    def __init__(self):
        self.train, self.test = self.load_and_divide()
        self.words, self.tags = self.get_words_and_tags()
        self.words_to_ind, self.tags_to_ind = self.word_and_tag_to_ind()
        self.num_features = len(self.tags)**2 + len(self.words)**2

    def word_and_tag_to_ind(self):
        """
        Creates 2 dictionaries that map word/tag to an index.
        :return: Tuple contains dictionary from word to index, and dictionary
         from tag to index.
        """
        word_to_ind_dict, tag_to_ind_dict = dict(), dict()
        words_length = len(self.words)
        for i in range(words_length):
            word_to_ind_dict[self.words[i]] = i
        tags_length = len(self.tags)
        for i in range(tags_length):
            tag_to_ind_dict[self.tags[i]] = i
        return word_to_ind_dict, tag_to_ind_dict

    def feature_func(self, u, v):
        """
        Creates the features for the poen(self.tags)**2 + len(self.words)**2tential arc between u and v.
        :param u: Node head of the arc.
        :param v: Node tail of the arc.
        :return: List contains features indexes of the potential arc.
        """
        w1, w2 = u[WORD_IND], v[WORD_IND]
        t1, t2 = u[TAG_IND], v[TAG_IND]
        features = [self.words_to_ind[w1] + (len(self.words) * self.words_to_ind[w2]),
                    len(self.words)**2 + self.tags_to_ind[t1] + (len(self.tags) * self.tags_to_ind[t2])]
        return features

    # === STATIC METHODS ===
    @staticmethod
    def load_and_divide():
        """
        Load the dependency treebank data and divide it into train and test.
        :return: Tuple contains the train data and the test data.
        """
        sents = dependency_treebank.parsed_sents()
        size = len(sents)
        train, test = sents[:int(size * SPLIT)], sents[int(size * SPLIT):]
        return train, test

    @staticmethod
    def get_words_and_tags():
        """
        Get all possible words and all possible tags in the data.
        :return: Tuple contains the possible words and the possible tags.
        """
        data = dependency_treebank.tagged_words()
        words, tags = zip(*data)
        words = ['ROOT'] + list(set(words))
        tags = ['ROOT'] + list(set(tags))
        return words, tags


class AveragedPerceptron:
    """
    Averaged Perceptron algorithm, as descriped in the lecture.
    """

    def __init__(self, features_dim, iterations, lr):
        self.features_dim = features_dim
        self.lr = lr
        self.iterations = iterations
        self.weights = np.zeros(features_dim)

    def get_arc_weight(self, arc):
        """
        Retrieves the score of the given arc. Equivalent as using dot product.
        :param arc: Acr object to obtain its score.
        :return: The score of the given arc, according to the current weights.
        """
        return self.weights[arc.features[0]] + self.weights[arc.features[1]]

    def update_weights(self, gt, mt):
        """
        Update weights vector by lr according to the location from features.
        :param gt: golden tree: List[Arc]
        :param mt: min tree: List[Arc]
        """
        gt_features = [arc.features for arc in gt]
        for arr in gt_features:
            for j in arr:
                self.weights[j] += self.lr

        mt_features = [arc.features for arc in mt]
        for arr in mt_features:
            for j in arr:
                self.weights[j] -= self.lr

    def train(self, x):
        """
        Train the model for number of iterations.
        :param x: Graphs to be trained on.
        """
        for i in range(self.iterations):
            print("iterartion", i)
            random.shuffle(x)  # Traverse in a random order to avoid artefacts
            for j, graph in enumerate(x):
                # Set scores for current graph
                for arc in graph.arcs:
                    arc.weight = self.get_arc_weight(arc)
                # Compute MST for current graph
                min_tree = max_spanning_arborescence_nx(graph.arcs, 0)
                # Update weights
                self.update_weights(graph.gold_tree, list(min_tree.values()))
                print("graph", j, " out of ", len(x))

        self.weights /= (self.iterations * len(x))  # Normalize

    def predict(self, x):
        """
        Predict the MST of graph x.
        :param x: Graph object.
        :return: The MST of x.
        """
        for arc in x.arcs:
            arc.weight = self.get_arc_weight(arc)
        return max_spanning_arborescence_nx(x.arcs, 0)

    @staticmethod
    def sent_acc(pred_tree, gold_tree, words_num):
        """
        Compute the (unlabeled) attachment score for the learned w,
         averaged over all sentences in the test set.
        :param gold_tree: The gold standard tree.
        :param pred_tree: The predicted tree.
        :param words_num: Number of words
        :return: Accuracy of the prediction of the model
        """
        shared_arcs = 0
        gold_arcs_set = {(arc.head, arc.tail) for arc in gold_tree}
        pred_arcs_set = {(arc.head, arc.tail) for arc in pred_tree}
        for arc in pred_arcs_set:
            if arc in gold_arcs_set:
                shared_arcs += 1
        return shared_arcs / (words_num - 1)


def min_spanning_arborescence_nx(arcs, sink):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = minimum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


def max_spanning_arborescence_nx(arcs, sink):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = maximum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


def main():
    nltk.download('dependency_treebank')
    data = DataContainer()
    model = AveragedPerceptron(data.num_features, ITERATIONS, LR)
    train_trees = [SentTree(sent.nodes, data) for sent in data.train]
    model.train(train_trees)
    test_trees = [SentTree(sent.nodes, data) for sent in data.test]
    accuracy = .0
    for i, tree in enumerate(test_trees):
        print("tree in test: ", i, " out of ", len(test_trees))
        accuracy += model.sent_acc(list(model.predict(tree).values()), tree.gold_tree, len(tree.nodes))

    accuracy /= len(test_trees)
    print('Accuracy over test set is {}'.format(accuracy))


if __name__ == '__main__':
    main()