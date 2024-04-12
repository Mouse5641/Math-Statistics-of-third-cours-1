import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def to_np_array(data_string):
    if data_string:
        data_array = np.array(json.loads(data_string))
        return data_array
    else:
        return None


def build_plot(data):
    plt.clf()
    x = data[:, 0]
    y = data[:, 1]
    # Побудова кореляційного поля
    plt.figure(figsize=(6.46, 6.46))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.scatter(x, y, alpha=0.5, c='b')
    plt.title('Двовимірний нормальний розподіл та кореляційне поле')

    x_ticks = [i for i in range(-5, 5)]  # Вкажіть власні значення
    y_ticks = [i for i in range(-5, 5)]  # Вкажіть власні значення
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.savefig("correlation_field.png")

    mean = [np.mean(x), np.mean(y)]  # Середнє значення
    cov_matrix = np.cov(data, rowvar=False)

    # Візуалізація функції щільності
    x, y = np.mgrid[-3:3:.01, -3:3:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov_matrix)
    plt.figure(figsize=(6.46, 6.46))
    plt.contourf(x, y, rv.pdf(pos), levels=100, cmap='viridis')
    plt.colorbar()
    plt.title('Функція щільності двовимірного нормального розподілу')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig("density_function.png")

    return "correlation_field.png", "density_function.png"
