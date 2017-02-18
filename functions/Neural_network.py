import sys
import time

import numpy as np

from functions.funct import Functions
from functions.pre_processing import PreProcess


class NeuralNetwork(object):
    def __init__(self, X_train, y_train, hidden_nodes, learning_rate, min_count, polarity_cutoff):
        self.review_vocab, self.word2index, self.label_vocab, self.label2index = \
            PreProcess(X_train, y_train, min_count, polarity_cutoff).return_pre_process()
        self.input_nodes = len(self.review_vocab)
        self.hidden_nodes = hidden_nodes
        self.output_nodes = 1
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes ** -0.5,
                                            (self.hidden_nodes, self.output_nodes))
        self.learning_rate = learning_rate
        self.layer_0 = np.zeros((1, self.input_nodes))
        self.layer_1 = np.zeros((1, hidden_nodes))
        self.reviews = X_train
        self.labels = y_train

    def train(self, training_reviews_raw, training_labels):
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        assert (len(training_reviews) == len(training_labels))
        correct_so_far = 0
        start = time.time()
        for i in range(len(training_reviews)):
            review = training_reviews[i]
            label = training_labels[i]
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            layer_2 = Functions.sigmoid(self.layer_1.dot(self.weights_1_2))
            layer_2_error = layer_2 - Functions.get_target_for_label(label)
            layer_2_delta = layer_2_error * Functions.sigmoid_output_2_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate
            if layer_2 >= 0.5 and label == 'POSITIVE':
                correct_so_far += 1
            if layer_2 < 0.5 and label == 'NEGATIVE':
                correct_so_far += 1
            reviews_per_second = i / float(time.time() - start)
            sys.stdout.write(
                "\rProgress:" + str(100 * i / float(len(training_reviews)))[:4]
                + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] +
                " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) +
                " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")

    def test(self, testing_reviews, testing_labels):
        correct = 0
        start = time.time()
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if pred == testing_labels[i]:
                correct += 1

            reviews_per_second = i / float(time.time() - start)

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4]
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] +
                             "% #Correct:" + str(correct) + " #Tested:" + str(i + 1) +
                             " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        # Output layer
        layer_2 = Functions.sigmoid(self.layer_1.dot(self.weights_1_2))

        if layer_2[0] >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"
