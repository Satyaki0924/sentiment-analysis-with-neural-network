import numpy as np


class Functions(object):
    def __init__(self):
        pass

    @staticmethod
    def update_input_layer(review, layer_0, word2index):
        layer_0 *= 0
        for word in review.split(" "):
            layer_0[0][word2index[word]] += 1
        return layer_0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_output_2_derivative(output):
        return output * (1 - output)

    @staticmethod
    def get_target_for_label(label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0
