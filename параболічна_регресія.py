import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy import stats


def parabolic_regression(data):
    regress_analys_data = []
    np_data = np.array(data)

    regress_analys_data.append("\n\n")
    regress_analys_data.append("Параболічний регресійний аналіз")

    X = np_data[:, 0]
    Y = np_data[:, 1]
    N = len(X)

    def parabolic_regress(x, a, b, c):
        return a * x ** 2 + b * x + c

    def calculate_phi_1(x, x_bar):
        return x - x_bar

    def calculate_phi_2(x, x_bar, x_squared_mean, x_cubed_mean, x_mean_squared):
        sigma_x_squared = x_squared_mean - x_bar ** 2
        return x ** 2 - ((x_cubed_mean - x_mean_squared * x_bar) / sigma_x_squared) * (x - x_bar) - x_mean_squared

    # Оцінки параметрів a, b, c
    params, covariance = curve_fit(parabolic_regress, X, Y)
    mean_x = sum(X) / N
    mean_y = sum(Y) / N

    a, b, c = params
    regress_analys_data.append(f"Оцінка параметра a: {a}")
    regress_analys_data.append(f"Оцінка параметра b: {b}")
    regress_analys_data.append(f"Оцінка параметра c: {c}")

    y_hat_values = parabolic_regress(X, a, b, c)

    x_squared_mean = np.mean(X ** 2)
    x_cubed_mean = np.mean(X ** 3)
    x_mean_squared = mean_x ** 2

    SE_y_hat = np.sqrt(np.sum((Y - y_hat_values) ** 2) / (N - 2))
    sigma_x_squared = np.sum((X - mean_x) ** 2) / (len(X) - 1)
    sigma_y_squared = np.sum((Y - mean_y) ** 2) / (len(Y) - 1)
    phi_1 = calculate_phi_1(X, mean_x)
    phi_2 = calculate_phi_2(X, mean_x, x_squared_mean, x_cubed_mean, x_mean_squared)
    Sxy = (SE_y_hat / np.sqrt(N)) * np.sqrt(1 + (phi_1 ** 2 / sigma_x_squared) + (phi_2 ** 2 / phi_2 ** 2))

    t_a = (a / SE_y_hat) * np.sqrt(N)
    t_b = ((b * sigma_x_squared) / SE_y_hat) * np.sqrt(N)
    t_c = (c / SE_y_hat) * np.sqrt(N * np.mean(phi_2 ** 2))

    regress_analys_data.append(f"t-статистика для параметра a: {t_a}")
    regress_analys_data.append(f"t-статистика для параметра b: {t_b}")
    regress_analys_data.append(f"t-статистика для параметра c: {t_c}")

    alpha = 0.05
    v = N - 3
    t_critical = t.ppf(1 - alpha / 2, df=v)

    if abs(t_a) >= t_critical:
        regress_analys_data.append("Оцінка параметра а є значущою")
    if abs(t_b) >= t_critical:
        regress_analys_data.append("Оцінка параметра b є значущою")
    if abs(t_c) >= t_critical:
        regress_analys_data.append("Оцінка параметра c є значущою")

    R = (1 - SE_y_hat ** 2 / sigma_y_squared) * 100
    regress_analys_data.append(f"Коефіцієнт детермінації: {R}%")

    f = SE_y_hat ** 2 / sigma_y_squared
    confidence_level = 0.95
    v1 = N - 1
    v2 = N - 3
    quantile = stats.f.ppf(confidence_level, v1, v2)

    if f <= quantile:
        regress_analys_data.append("Запропонована регресійна залежність є значуща")
    else:
        regress_analys_data.append("Запропонована регресійна залежність не є значуща")

    confidence_interval_lower = y_hat_values - t_critical * Sxy
    confidence_interval_upper = y_hat_values + t_critical * Sxy

    tolerance_interval_lower = c + b * phi_1 + a * phi_2 - t_critical * SE_y_hat
    tolerance_interval_upper = c + b * phi_1 + a * phi_2 + t_critical * SE_y_hat

    plt.clf()
    plt.scatter(X, Y, label='Дані')
    plt.plot(X, y_hat_values, color='red', label='Лінія регресії')
    plt.plot(X, confidence_interval_lower, '--', color='green', label='Довірчий інтервал')
    plt.plot(X, confidence_interval_upper, '--', color='green')
    plt.plot(X, tolerance_interval_lower, '--', color='blue', label='Толерантні межі')
    plt.plot(X, tolerance_interval_upper, '--', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("parabolic_regression.png")

    return "\n".join(regress_analys_data), "parabolic_regression.png"
