import scipy.stats as stats
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import norm


def correlation_coefficient(data):
    list1 = []
    np_data = np.array(data)

    X = np_data[:, 0]
    Y = np_data[:, 1]
    N = len(X)

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    sigma_x_squared = np.sum((X - mean_x) ** 2) / (len(X) - 1)
    sigma_y_squared = np.sum((Y - mean_y) ** 2) / (len(Y) - 1)

    result = np.dot(X, Y)
    xy = (1 / N) * result
    r = (N / (N - 1)) * ((xy - mean_x * mean_y) / (np.std(X) * np.std(Y)))

    first = (1 / (2 * np.pi * sigma_x_squared * sigma_y_squared * np.sqrt(1 - r ** 2)))
    second = (((X - mean_x) / sigma_x_squared) - (
            2 * r * ((X - mean_x) / sigma_x_squared) * ((Y - mean_y) / sigma_y_squared)) + (
                      (Y - mean_y) / sigma_y_squared) ** 2)
    third = np.exp(-(1 / (2 * (1 - r ** 2))) * second)
    f_x_y = first * third

    mean_f = np.mean(f_x_y)
    hx = (np.max(X) - np.min(X)) / N
    hy = (np.max(Y) - np.min(Y)) / N
    p = mean_f * hx * hy

    reproduced_relative_frequencies = {value: np.sum(X == value) / N for value in set(X)}
    frequencies = list(reproduced_relative_frequencies.values())

    xi_x = np.sum((frequencies - p) ** 2 / p)
    # xi_y = np.sum((frequencies - p) ** 2 / p)
    # xi_x_y = np.sum(np.sum((frequencies - p) ** 2 / p))

    df = N - 1
    alpha = 0.05
    critical_value = stats.chi2.ppf(1 - alpha, df)

    list1.append(f"xi-квадратт за змінною y при фіксованій x: {xi_x}")
    # list.append(f"xi-квадратт за змінною x при фіксованій y: {xi_y}")
    # list.append(f"xi-квадратт одночасно за змінними x та y: {xi_x_y}")

    if xi_x > critical_value:
        list1.append(f"Відхиляємо нульову гіпотезу: Модель не відтворює розподіл.")
    else:
        list1.append(f"Не відхиляємо нульову гіпотезу: Модель відтворює розподіл.")

    result = np.dot(X, Y)
    xy = (1 / N) * result

    r = (N / (N - 1)) * ((xy - mean_x * mean_y) / (sigma_x_squared * sigma_y_squared))

    _, t_test = stats.pearsonr(X, Y)
    list1.append(f"Оцінка коефіцієнту кореляції: {r}")
    # list1.append(f"T-тест: {t_test}")

    alpha = 0.05
    if t_test < alpha:
        list1.append("Кореляція є статистично значущою")
    else:
        list1.append("Кореляція не є статистично значущою")

    n = len(X)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci = r_z + np.array([-1, 1]) * z_critical * se
    ci = np.tanh(ci)
    list1.append(f"Довірчий інтервал: {ci}")

    list1.append("")

    overall_mean = np.mean(data)
    # Обчислення загальної дисперсії
    overall_variance = np.mean((data - overall_mean) ** 2)
    # Обчислення міжгрупової дисперсії
    between_group_variance = np.mean((np.mean(X) - overall_mean) ** 2) + np.mean(
        (np.mean(Y) - overall_mean) ** 2)
    ratio = between_group_variance / overall_variance
    list1.append(f"Коефіцієнт кореляційного відношення: {ratio}")

    v1 = 1
    v2 = len(X) - 2
    # Обчислення статистики f
    f = ((r ** 2) / (1 - r ** 2) * (v2 / v1))
    # Визначення критичного значення f
    alpha = 0.05
    f_critical = stats.f.ppf(1 - alpha, v1, v2)
    if f > f_critical:
        list1.append("Кореляційне відношення значуще")
    else:
        list1.append("Кореляційне відношення не значуще")

    list1.append("")

    #обчислення рангового коефіцієнта кореляції Спірмена
    rho, p_value = spearmanr(X, Y)
    list1.append(f'Ранговий коефіцієнт кореляції Спірмена: {rho}')
    alpha = 0.05
    if p_value < alpha:
        list1.append("Результат є статистично значущим.")
    else:
        list1.append("Результат не є статистично значущим.")

    list1.append("")

    tau, p_value = kendalltau(X, Y)
    list1.append(f'Ранговий коефіцієнт Кендалла: {tau}')
    alpha = 0.05
    if p_value < alpha:
        list1.append("Результат є статистично значущим.")
    else:
        list1.append("Результат не є статистично значущим.")

    X1 = np.zeros(N)
    Y1 = np.zeros(N)

    for i in range(N):
        if X[i] - mean_x <= 0:
            X1[i] = 0
        else:
            X1[i] = 1

        if Y[i] - mean_y <= 0:
            Y1[i] = 0
        else:
            Y1[i] = 1

    n00 = 1
    n01 = 1
    n10 = 1
    n11 = 1

    for i in range(N):
        if X1[i] == 0 and Y1[i] == 0:
            n00 += 1
        elif X1[i] == 0 and Y1[i] == 1:
            n10 += 1
        elif X1[i] == 1 and Y1[i] == 0:
            n01 += 1
        elif X1[i] == 1 and Y1[i] == 1:
            n11 += 1

    cross_table_new = np.array([[n00, n01, 0],
                                [n10, n11, 0],
                                [0, 0, 0]])

    cross_table_new[0][2] = n00 + n01
    cross_table_new[1][2] = n10 + n11
    cross_table_new[2][0] = n00 + n10
    cross_table_new[2][1] = n01 + n11
    cross_table_new[2][2] = cross_table_new[0][2] + cross_table_new[1][2]

    # list1.append(f"Таблиця сполучень 2х2:")
    # list1.append(cross_table_new)

    # індекс Фехнера
    I = (cross_table_new[0][0] + cross_table_new[1][1] - cross_table_new[1][0] - cross_table_new[0][1]) / (
            cross_table_new[0][0] + cross_table_new[1][1] + cross_table_new[1][0] + cross_table_new[0][1])

    # Коефіцієнт сполучень («Фі»),
    F = (cross_table_new[0][0] * cross_table_new[1][1] - cross_table_new[1][0] * cross_table_new[0][1]) / np.sqrt(
        cross_table_new[0][2] * cross_table_new[1][2] * cross_table_new[2][0] * cross_table_new[2][1])

    # Коефіцієнти зв’язку Юла
    Q = (cross_table_new[0][0] * cross_table_new[1][1] - cross_table_new[1][0] * cross_table_new[0][1]) / (
            cross_table_new[0][0] * cross_table_new[1][1] + cross_table_new[1][0] * cross_table_new[0][1])
    Y_k = (np.sqrt(cross_table_new[0][0] * cross_table_new[1][1]) - np.sqrt(
        cross_table_new[1][0] * cross_table_new[0][1])) / (
                  np.sqrt(cross_table_new[0][0] * cross_table_new[1][1]) + np.sqrt(
              cross_table_new[1][0] * cross_table_new[0][1]))
    # Q= (2*Y)/(1+Y)**2

    list1.append(f"Індекс Фехнера: {I}")
    if I > 0:
        list1.append("Додатна кореляція")
    elif I < 0:
        list1.append("Від'ємна кореляція")
    elif I == 0:
        list1.append("X та У - незалежні")
    list1.append("")
    list1.append(f"Коефіцієнт сполучень («Фі»): {F}")

    df = 1
    alpha = 0.05
    critical_value = stats.chi2.ppf(1 - alpha, df)
    if N >= 40 and n00 >= 5 and n01 >= 5 and n10 >= 5 and n11 >= 5:
        xi = N * F ** 2
        if xi >= critical_value:
            list1.append("Оцінка коефіцієнта Фі є значущою")
        else:
            list1.append("Оцінка коефіцієнта Фі не є значущою")
    else:
        xi = N * (n00 * n11 - n01 * n10 - 0.5) ** 2 / (
                cross_table_new[0][2] * cross_table_new[1][2] * cross_table_new[2][1] * cross_table_new[2][2])
        if xi >= critical_value:
            list1.append("Оцінка коефіцієнта Фі є значущою")
        else:
            list1.append("Оцінка коефіцієнта Фі не є значущою")
    list1.append("")

    list1.append(f"Коефіцієнти зв’язку Юла: Q={Q} Y={Y_k}")
    S_q = 0.5 * (1 - Q ** 2) * np.sqrt(1 / n00 + 1 / n01 + 1 / n10 + 1 / n11)
    S_y = 0.25 * (1 - Y_k ** 2) * np.sqrt(1 / n00 + 1 / n01 + 1 / n10 + 1 / n11)

    u_q = Q / S_q
    u_y = Y_k / S_y

    a = 0.05
    u1 = norm.ppf(1 - a / 2)
    result = u1 - a / 2

    if u_q <= result and u_y <= result:
        list1.append("Коефіцієнти не є значущим")
    else:
        list1.append("Коефіцієнт є значущим")

    pearson_corr, p_value = stats.pearsonr(X, Y)
    list1.append(f"Коефіцієнт кореляції Пірсона: {pearson_corr}")
    if pearson_corr < p_value:
        list1.append("Не є статистично значущим")
    else:
        list1.append("Є статистично значущим")

    if len(X) == len(Y):
        kendall_corr, p_value = stats.kendalltau(X, Y)
        list1.append(f"Коефіцієнт кореляції Кендалла: {kendall_corr}")
        if kendall_corr < p_value:
            list1.append("Не є статистично значущим")
        else:
            list1.append("Є статистично значущим")

    else:
        spearman_corr, _ = stats.spearmanr(X, Y)
        list1.append(f"Коефіцієнт кореляції Спірмена (Стюарта): {spearman_corr}")
        if spearman_corr < p_value:
            list1.append("Не є статистично значущим")
        else:
            list1.append("Є статистично значущим")

    return "\n".join(list1)