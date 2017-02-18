import sys
import time
from collections import Counter
from string import punctuation

import numpy as np
from nltk.corpus import stopwords


class PreProcess(object):
    def __init__(self, X_train, y_train, min_count, polarity_cutoff):
        self.reviews = X_train
        self.labels = y_train
        self.review_vocab = list()
        self.label_vocab = list()
        self.word2index = {}
        self.label2index = {}
        self.pos_neg_ratio = Counter()
        self.total_counts = Counter()
        self.min_count = min_count
        self.polarity_cutoff = polarity_cutoff
        self.preprocess_data()

    def preprocess_data(self):
        print('Author: Satyaki Sanyal')
        sys.stdout.write("\rStarting with pre-processing...")
        positive_counts = Counter()
        negative_counts = Counter()
        for i in range(len(self.reviews)):
            if self.labels[i] == 'POSITIVE':
                for words in self.reviews[i].split(" "):
                    positive_counts[words] += 1
                    self.total_counts[words] += 1
            else:
                for words in self.reviews[i].split(" "):
                    negative_counts[words] += 1
                    self.total_counts[words] += 1
        for term, count in list(self.total_counts.most_common()):
            if count >= 50:
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
                self.pos_neg_ratio[term] = pos_neg_ratio
        for word, ratio in self.pos_neg_ratio.most_common():
            if ratio > 1:
                self.pos_neg_ratio[word] = np.log(ratio)
            else:
                self.pos_neg_ratio[word] = -np.log((1 / (0.01 + ratio)))
        self.read_reviews()

    def read_reviews(self):
        review_vocab = set()
        start = time.time()
        for count, review in enumerate(self.reviews):
            for word in review.split(" "):
                if self.total_counts[word] > self.min_count:
                    if word in self.pos_neg_ratio.keys():
                        if self.pos_neg_ratio[word] >= self.polarity_cutoff \
                                or self.pos_neg_ratio[word] <= - self.polarity_cutoff:
                            review_vocab.add(word)
            reviews_per_second = count / float(time.time() - start)
            sys.stdout.write("\rReading training reviews. Progress : " +
                             str((count * 100) / len(self.reviews))[: 4] + " %" +
                             "\t word counting : " + str(len(review_vocab)) +
                             "\t speed (reviews/sec) : " + str(reviews_per_second)[: 4])
        self.process_review(review_vocab)

    def process_review(self, review_vocab):
        review_vocab = list(review_vocab)
        start = time.time()
        for count, word in enumerate(review_vocab):
            reviews_per_second = count / float(time.time() - start)
            if word not in stopwords.words('english') and word not in punctuation:
                self.review_vocab.append(word)
            sys.stdout.write("\rPre-processing training reviews. Progress : " +
                             str((count * 100) / len(review_vocab))[: 4] + " %" +
                             "\t word processing : " + str(len(self.review_vocab)) +
                             "\t speed (words/sec) : " + str(reviews_per_second)[: 4])
        self.process_label()

    def process_label(self):
        label_vocab = set()
        start = time.time()
        for count, label in enumerate(self.labels):
            labels_per_second = count / float(time.time() - start)
            label_vocab.add(label)
            sys.stdout.write("\rPre-processing labels. Progress : " +
                             str((count * 100) / len(self.labels))[: 4] + " %" +
                             "\t word processing : " + str(len(label_vocab)) +
                             "\t speed (labels/sec) : " + str(labels_per_second)[: 4])
        self.label_vocab.extend(list(label_vocab))
        self.index_word()

    def index_word(self):
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        self.index_label()

    def index_label(self):
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def return_pre_process(self):
        sys.stdout.write("\rPre-processing completed...")
        return self.review_vocab, self.word2index, self.label_vocab, self.label2index
