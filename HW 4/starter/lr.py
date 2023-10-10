import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    input_file = str(sys.argv[1])
    # print(input_file)
    output_file = str(sys.argv[2])
    # print(output_file)
    features, label = read_data(input_file)

    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    iteration = 100
    # print(features, label)
    weight_list = []
    for cur_lr in lr_list:
        # final_weights = lr(cur_lr, features, label, iteration, loss=True)
        final_weights = lr(cur_lr, features, label, iteration, loss=False)
        weight_list.append(final_weights)

    my_lr = 0.8
    my_iteration = 30
    # final_weights = lr(my_lr, features, label, my_iteration, loss=True)
    final_weights = lr(my_lr, features, label, my_iteration, loss=False)

    weight_list.append(final_weights)

    lr_array = np.array([[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.8]]).reshape([10, 1])

    iteration_array = np.array([[100, 100, 100, 100, 100, 100, 100, 100, 100, 30]]).reshape([10, 1])

    weight_array = np.asarray(weight_list)
    output_array = np.hstack([lr_array, iteration_array, weight_array[:, 0, :]])
    pd.DataFrame(output_array).to_csv(output_file,
                                      index=False, header=False)


def lr(learning_rate, features, label, iteration, loss=False):
    weights = np.zeros([1, features.shape[1]])
    if loss:
        loss_list = []
    for i in range(iteration):
        fx_list = []
        for j in range(features.shape[0]):
            cur_fx = np.sum(weights @ features[j])
            fx_list.append(cur_fx)
        # print("fxlist", fx_list)
        if loss:
            loss_list.append(cal_loss(features, fx_list, label))
        grad = cal_grad(fx_list, features, label)
        # print(grad.shape)
        weights -= learning_rate * grad

        # _loss = loss(features, fx_list, label)
    # print(weights)
    if loss:
        plt.figure()
        plt.plot(loss_list)
        plt.yscale('log', base=20)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Loss')
        plt.title("Learning Rate: " + str(learning_rate))
        plt.show()
    return weights


def cal_grad(fx, features, label):
    n = len(fx)
    grad_list = []
    for i in range(features.shape[1]):
        cur_grad = 0
        for j in range(features.shape[0]):
            cur_grad += (fx[j] - label[j]) * features[j, i]
        grad_list.append(cur_grad / n)

    return np.asarray(grad_list)


def cal_loss(feature, fx, label):
    loss = 0
    for i in range(feature.shape[0]):
        loss += np.power(label[i] - fx[i], 2)
    loss = loss / 2 / feature.shape[0]
    return loss


def read_data(filename):
    dataframe = pd.read_csv(filename, names=["age", "weight", "height"])

    # rescale the data
    age_mean, age_std = dataframe["age"].mean(), dataframe["age"].std()
    weight_mean, weight_std = dataframe["weight"].mean(), dataframe["weight"].std()
    # print(age_mean, age_std, weight_mean, weight_std)
    dataframe["age"] = (dataframe["age"] - age_mean) / age_std
    dataframe["weight"] = (dataframe["weight"] - weight_mean) / weight_std

    all_data = dataframe.to_numpy()
    features = np.hstack([np.ones([all_data.shape[0], 1]), all_data[:, 0: -1]])
    label = all_data[:, -1]
    return features, label


if __name__ == "__main__":
    main()
