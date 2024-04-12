import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy import stats


def quasi_linear_regression(data):
    regress_analys_data = []
    np_data = np.array(data)

    X = np_data[:, 0]
    Y = np_data[:, 1]
    N = len(X)

    regress_analys_data.append("\n\n")
    regress_analys_data.append("Квазілінійний регресійний аналіз")

    def regression_function(x, a, b):
        return a + b * np.log(x)

    params, covariance = curve_fit(regression_function, X, Y)
    a, b = params

    regress_analys_data.append(f"Оцінка параметра a: {a}")
    regress_analys_data.append(f"Оцінка параметра b: {b}")

    y_values_for_line = regression_function(X, a, b)

    tt = np.log(X)
    mean_y = sum(Y) / N
    mean_t = sum(tt) / N
    z = y_values_for_line

    sigma_y_squared = np.sum((Y - mean_y) ** 2) / (len(Y) - 1)
    sigma_t_squared = np.sum((tt - mean_t) ** 2) / (len(Y) - 1)

    S_zal = np.sum((y_values_for_line - np.mean(z)) ** 2) / (len(Y) - 2)
    S_zal2 = np.sum((Y - y_values_for_line) ** 2) / (N - 2)

    S_a = np.sqrt(S_zal) * np.sqrt((1 / N) + (mean_t ** 2 / sigma_t_squared * (N - 1)))
    S_b = np.sqrt(S_zal) / (np.sqrt(sigma_t_squared) * np.sqrt(N - 1))

    S_z = np.sqrt(S_zal * (1 / N) + S_b ** 2 * (tt - mean_t) ** 2)

    alpha = 0.05
    v = N - 2
    t_critical = t.ppf(1 - alpha / 2, df=v)

    t_a = a / S_a
    t_b = b / S_b

    regress_analys_data.append(f"t-статистика для параметра a: {t_a}")
    regress_analys_data.append(f"t-статистика для параметра b: {t_b}")

    if np.abs(t_a) > t_critical:
        regress_analys_data.append("Параметр a є значущим")
    else:
        regress_analys_data.append("Параметр a не є значущим")

    if np.abs(t_b) > t_critical:
        regress_analys_data.append("Параметр b є значущим")
    else:
        regress_analys_data.append("Параметр b не є значущим")

    z_min = z - t_critical * S_z
    z_max = z + t_critical * S_z

    a_min = z - t_critical * np.sqrt(S_zal2)
    a_max = z + t_critical * np.sqrt(S_zal2)

    r_squared = (1 - np.sum((y_values_for_line - Y) ** 2) / np.sum((Y - mean_y) ** 2)) * 100
    regress_analys_data.append(f"Коефіцієнт детермінації: {r_squared}%")

    f = S_zal / sigma_y_squared
    confidence_level = 0.95
    v1 = N - 1
    v2 = N - 3
    quantile = stats.f.ppf(confidence_level, v1, v2)

    if f <= quantile:
        regress_analys_data.append("Запропонована регресійна залежність є значуща")
    else:
        regress_analys_data.append("Запропонована регресійна залежність не є значуща")

    plt.clf()
    plt.scatter(X, Y, label='Дані')
    plt.plot(X, y_values_for_line, color='red', label='Лінія регресії')

    plt.plot(X, z_min, '--', color='green', label='Довірчий інтервал')
    plt.plot(X, z_max, '--', color='green')

    plt.plot(X, a_min, '--', color='black', label='Толерантні межі')
    plt.plot(X, a_max, '--', color='black')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("quasi_linear_regression.png")

    return "\n".join(regress_analys_data), "quasi_linear_regression.png"