import numpy as np
import matplotlib.pyplot as plt


def explore_visual(model_params, explore_errors):
    x = range(1360)

    fig = plt.figure()
    plt.scatter(x[:16], explore_errors[:16], c='b', marker='o', label='1 layer')
    plt.scatter(x[16:80], explore_errors[16:80], c='r', marker='o', label='2 layer')
    plt.scatter(x[80:336], explore_errors[80:336], c='g', marker='o', label='3 layer')
    plt.scatter(x[336:1360], explore_errors[336:1360], c='m', marker='o', label='4 layer')
    plt.legend(loc='upper left')
    plt.show()

    min_model_index = np.argmin(np.array(explore_errors))
    print("Min. model index: ", min_model_index)
    print("Min. model params: ", model_params[min_model_index])
    print("Min. error: ", explore_errors[min_model_index])
