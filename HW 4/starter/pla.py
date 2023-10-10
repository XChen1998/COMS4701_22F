import sys

import numpy as np
import pandas as pd


def main():
    '''YOUR CODE GOES HERE'''
    input_file = str(sys.argv[1])
    # print(input_file)
    output_file = str(sys.argv[2])
    # print(output_file)
    features, label = read_data(input_file)
    # print(features)
    # print(features, label)
    weights_history = perceptron(features, label)
    pd.DataFrame(weights_history[:, 0, :]).to_csv(output_file,
                                                  index=False, header=False)


def read_data(filename):
    dataframe = pd.read_csv(filename, names=["x1", "x2", "y"])
    # print(dataframe)
    all_data = dataframe.to_numpy()
    features = all_data[:, 0: -1]
    label = all_data[:, -1]
    return features, label


def perceptron(features, label):
    bias_feature = np.ones((features.shape[0], 1))
    new_features = np.hstack([features, bias_feature])
    number_of_points = new_features.shape[0]
    weights = np.zeros((1, new_features.shape[1]))
    end = False
    weights_history = []
    # print(new_features)
    while not end:
        end = True
        # print(weights)
        weights_history.append(np.array(weights))
        for i in range(number_of_points):
            # print(weights.shape)
            # print(new_features[i].shape)
            kernel = np.sum(weights * new_features[i])
            # print(kernel)
            sign = -1 if kernel <= 0 else 1
            if sign * label[i] < 0:
                weights += label[i] * new_features[i]
                end = False
    weights_history.append(np.array(weights))
    weights_history = np.asarray(weights_history)
    # print(weights_history)
    return np.int64(weights_history)


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
