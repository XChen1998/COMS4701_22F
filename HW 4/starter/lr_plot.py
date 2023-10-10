import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


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
    features, label, df = read_data(input_file)

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
    for i in range(lr_array.shape[0]):

        visualize_3d(df, lin_reg_weights = weight_array[i][0], feat1="age", feat2="weight", labels="height", xlim=(-3, 3),
                     ylim=(-2, 2), zlim=(0, 3), alpha=lr_array[i], xlabel='Normalised Age', ylabel='Normalised Weight', zlabel='Height (m)', title='')

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
    return features, label, dataframe

def visualize_3d(df, lin_reg_weights=[1, 1, 1], feat1=0, feat2=1, labels=2,
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=''):
    """
    3D surface plot.
    Main args:
      - df: dataframe with feat1, feat2, and labels
      - feat1: int/string column name of first feature
      - feat2: int/string column name of second feature
      - labels: int/string column name of labels
      - lin_reg_weights: [b_0, b_1 , b_2] list of float weights in order
    Optional args:
      - x,y,zlim: axes boundaries. Default to -1 to 1 normalized feature values.
      - alpha: step size of this model, for title only
      - x,y,z labels: for display only
      - title: title of plot
    """

    # Setup 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = plt.figure().gca(projection='3d')
    # plt.hold(True)

    # Add scatter plot
    ax.scatter(df[feat1], df[feat2], df[labels])
    print(df[feat1])
    print(df[feat2])
    print(df[labels])
    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array([lin_reg_weights[0] +
                      lin_reg_weights[1] * f1 +
                      lin_reg_weights[2] * f2  # height
                      for f1, f2 in zip(axes1, axes2)])
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1)
    ax.view_init(5, 45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    if title == '':
        title = 'LinReg Height with Alpha %f' % alpha
    ax.set_title(title)

    plt.show()

if __name__ == "__main__":
    main()
