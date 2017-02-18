import os


class ReadData(object):
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.reviews = None
        self.labels = None
        self.read_reviews()
        self.read_labels()

    def read_reviews(self):
        with open(self.path + '/../data/reviews.txt', 'r') as file:
            self.reviews = list(map(lambda x: x[:-1], file.readlines()))

    def read_labels(self):
        with open(self.path + '/../data/labels.txt', 'r') as file:
            self.labels = list(map(lambda x: x[:-1].upper(), file.readlines()))

    def return_value(self):
        return self.reviews, self.labels
