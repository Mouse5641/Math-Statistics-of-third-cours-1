from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def linear_regression(data):
    regress_analys_data = []
    np_data = np.array(data)

    regress_analys_data.append("\n\n")
    regress_analys_data.append("Лінійний регресійний аналіз")

    X = np_data[:, 0]
    Y = np_data[:, 1]
    N = len(X)

    correlation, p_value = stats.pearsonr(X, Y)

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    sigma_x_squared = np.sum((X - mean_x) ** 2) / (len(X) - 1)
    sigma_y_squared = np.sum((Y - mean_y) ** 2) / (len(Y) - 1)

    b = correlation * (np.sqrt(sigma_y_squared) / np.sqrt(sigma_x_squared))
    a = mean_y - b * mean_x

    regress_analys_data.append(f"Оцінка параметра a: {a}")
    regress_analys_data.append(f"Оцінка параметра b: {b}")
    Y_pred = b * X + a

    S_zal = np.sqrt(np.mean((Y - (a + b * X)) ** 2) / (N - 2))

    S_a = S_zal * np.sqrt(1 / N + mean_x ** 2 / (sigma_x_squared * (N - 1)))
    S_b = S_zal / (np.sqrt(sigma_x_squared) * np.sqrt(N - 1))

    t_a = a / S_a
    t_b = b / S_b

    alpha = 0.05
    v = N - 2

    t_critical = t.ppf(1 - alpha / 2, df=v)

    if t_b <= t_critical:
        regress_analys_data.append("Регресійний зв'язок не є значущим.")
    else:
        regress_analys_data.append("Регресійний зв'язок є значущим.")

    R = (1 - S_zal ** 2 / sigma_y_squared) * 100
    regress_analys_data.append(f"Коефіцієнт детермінації: {R}%")

    tolerance_interval_min = Y_pred - t_critical * S_zal
    tolerance_interval_max = Y_pred + t_critical * S_zal

    S_Y_bar = np.sqrt(S_zal ** 2 * 1 / N + S_b ** 2 * (X - mean_x) ** 2)

    confidence_interval_min = Y_pred - t_critical * S_Y_bar
    confidence_interval_max = Y_pred + t_critical * S_Y_bar

    std_errors = np.sqrt(S_zal ** 2 * (1 + (1 / N)) + S_b ** 2 * (X - mean_x) ** 2)

    lower_bounds = Y_pred - t_critical * std_errors
    upper_bounds = Y_pred + t_critical * std_errors

    f = S_zal ** 2 / sigma_y_squared
    confidence_level = 0.95
    v1 = N - 1
    v2 = N - 3
    quantile = stats.f.ppf(confidence_level, v1, v2)

    if f <= quantile:
        regress_analys_data.append("Запропонована регресійна залежність є значуща")
    else:
        regress_analys_data.append("Запропонована регресійна залежність не є значуща")

    plt.clf()
    plt.scatter(X, Y, label='Спостереження')
    plt.plot(X, Y_pred, color='red', label='Регресійна лінія')
    plt.fill_between(X, tolerance_interval_min, tolerance_interval_max, color='blue', alpha=0.5,
                     label='Tolerance Intervals')
    plt.fill_between(X, confidence_interval_min, confidence_interval_max, color='green', alpha=0.3,
                     label='Довірчий інтервал')
    plt.fill_between(X, lower_bounds, upper_bounds, color='yellow', alpha=0.3, label='для прогнозу нового спостереження', linewidth=4)
    plt.xlabel('Незалежна змінна (X)')
    plt.ylabel('Залежна змінна (Y)')
    plt.legend()
    plt.savefig("linear_regression.png")

    return "\n".join(regress_analys_data), "linear_regression.png"
