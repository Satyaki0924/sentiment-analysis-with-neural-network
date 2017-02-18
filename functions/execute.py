import sys

from sklearn.model_selection import train_test_split

from functions.Neural_network import NeuralNetwork
from functions.read_data import ReadData


class SentimentNetwork(object):
    def __init__(self, choice, test_size=0.2, min_count=20, polarity_cutoff=0.05, hidden_nodes=30, learning_rate=0.01):
        self.reviews = None
        self.labels = None
        self.test_size = test_size
        if choice == 1:
            self.test_data(hidden_nodes, learning_rate, min_count, polarity_cutoff)
        else:
            self.run_data(hidden_nodes, learning_rate, min_count, polarity_cutoff)

    def read_data(self):
        self.reviews, self.labels = ReadData().return_value()

    def test_data(self, hidden_nodes, learning_rate, min_count, polarity_cutoff):
        try:
            self.read_data()
            X_train, X_test, y_train, y_test = \
                train_test_split(self.reviews, self.labels, test_size=self.test_size, random_state=42)
            classifier = NeuralNetwork(X_train, y_train, hidden_nodes, learning_rate, min_count, polarity_cutoff)
            classifier.train(X_train, y_train)
            classifier.test(X_test, y_test)
        except Exception as e:
            print('Your system faced an exception: ' + str(e))

    def run_data(self, hidden_nodes, learning_rate, min_count, polarity_cutoff):
        try:
            self.read_data()
            classifier = NeuralNetwork(self.reviews, self.labels, hidden_nodes,
                                       learning_rate, min_count, polarity_cutoff)
            classifier.train(self.reviews, self.labels)
            sys.stdout.write("\r Enter your text after '>>' to analyse sentiments... Press 'ctrl + D' to break")
            print('')
            while True:
                try:
                    sentiment = input('>>\t')
                    print(classifier.run(sentiment) + '\n')
                except:
                    break
        except Exception as e:
            print('Your system faced an exception: ' + str(e))

