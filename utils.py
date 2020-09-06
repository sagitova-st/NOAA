from types import ArrayFloat
import numpy as np


def func_indexed_params(func, index, *kargs):
    '''
    Служебная функция для передачи только той части вектора, для которой нужно провести вычисления.
    :param func: функция, которая будет вызвана
    :param index: массив boolean, где True указывает на элементы, которые должны участвовать в вычислениях
    :param *kargs: аргументы вызываемой функции. Должны совпадать по количеству с ожидаемым функцией
                    Каждый аргумент должен быть массивом того же размера, что и index, или скаляром.
                    Скаляры передаются без изменения.
    :return: значение, возвращаемое функцией. Зависит от конкретной func. Можно ожидать, что размеры
             результата будут определяться количеством значений True в index

    Согласованность размера с index означает, что если index N-мерный массив,
    то первые N измерений другого массива должны совпадать с размерами соответствующих измерений index
    '''
    if not np.any(index):
        return []
    args = [0] * len(kargs)
    for i, arg in enumerate(kargs):
        if arg.shape[0] == len(index):
            args[i] = arg[index]
        else:
            args[i] = arg
    return func(*args)


def func_indexed_params_save_res(res, func, index, *kargs) -> None:
    '''
    Служебная функция для передачи только той части вектора, для которой нужно провести вычисления
    и записи результата в res
    :param res: массив для сохранения результатов. Обновляются только элементы для которых в index установлено True
                Вызывающая сторона контроллирует, что размер совместим с index и результатом возвращаемым func
    :param func: функция, которая будет вызвана
    :param index: массив boolean, где True указывает на элементы, которые должны участвовать в вычислениях
    :param *kargs: аргументы вызываемой функции. Должны совпадать по количеству с ожидаемым функцией
                    Каждый аргумент должен быть массивом того же размера, что и index, или скаляром.
                    Скаляры передаются без изменения.
    :rtype: None

    Согласованность размера с index означает, что если index N-мерный массив,
        то первые N измерений другого массива должны совпадать с размерами соответствующих измерений index
    '''
    res[index] = func_indexed_params(func, index, *kargs)


def true_anomaly_to_mean_anomaly_eccentric(true_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из истинной аномалии в среднюю. Перевод осуществляется через эксцентрическую аномалию.

    :param true_anomaly: вектор истинных аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор средних аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    eccentric_anomaly = true_anomaly_to_eccentric_anomaly(true_anomaly, eccentricity)
    mean_anomaly = eccentric_anomaly_to_mean_anomaly(eccentric_anomaly, eccentricity)
    return mean_anomaly


def mean_anomaly_to_true_anomaly_eccentric(mean_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из средней аномалии в истинную. Перевод осуществляется через эксцентрическую аномалию.

    :param mean_anomaly: вектор средних аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор истинных аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    eccentric_anomaly = mean_anomaly_to_eccentric_anomaly(mean_anomaly, eccentricity)
    true_anomaly = eccentric_anomaly_to_true_anomaly(eccentric_anomaly, eccentricity)
    return true_anomaly


def mean_anomaly_to_eccentric_anomaly(mean_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из средней аномалии в эксцентрическую.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a, p. 76 section 3.3.1
    :param mean_anomaly: вектор средних аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор эксцентрических аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''

    # pylint: disable=invalid-name
    E_res = mean_anomaly + np.where(np.abs(mean_anomaly) > np.pi, -1, 1) * np.sign(mean_anomaly)
    index = E_res > E_res - 1
    while np.any(index):
        E = E_res[index]  # pylint: disable=invalid-name

        # pylint: disable=invalid-name
        E_next = E + (mean_anomaly[index] - E + eccentricity[index] * np.sin(E)) / (1 - eccentricity[index] * np.cos(E))

        need_computation = np.abs(E_next - E) > 1e-8
        E_res[index] = E_next
        index[index] = need_computation
    return E_res


def eccentric_anomaly_to_mean_anomaly(eccentric_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из эксцентрической аномалии в среднюю.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a, p. 67 (3.137)
    :param eccentric_anomaly: вектор эксцентрических аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор истинных аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    return eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)


def true_anomaly_to_eccentric_anomaly(true_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из истинной аномалии в эксцентрическую.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a, p. 63 (3.116)
    Учтено, что :math:`1 + e cos(\\nu) > 0` для эллиптических орбит и не меняет квадрант.
    :param true_anomaly: вектор истинных аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор эксцентрических аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    pre_sin = np.sqrt(1 - eccentricity * eccentricity) * np.sin(true_anomaly)
    pre_cos = eccentricity + np.cos(true_anomaly)
    return np.arctan2(pre_sin, pre_cos)
    # return 2 * np.arctan2(np.tan(true_anomaly / 2), np.sqrt((1 + eccentricity) / (1 - eccentricity)))


def eccentric_anomaly_to_true_anomaly(eccentric_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из эксцентрической аномалии в истинную.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a, p. 77 (3.208)
    Учтено, что :math:`1 - e cos(E) > 0` для эллиптических орбит и не меняет квадрант.
    :param eccentric_anomaly: вектор эксцентрических аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор истинных аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    return np.arctan2(np.sqrt(1 - eccentricity * eccentricity) * np.sin(eccentric_anomaly),
                      np.cos(eccentric_anomaly) - eccentricity)


def true_anomaly_to_mean_anomaly_hyperbolic(true_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из истинной аномалии в среднюю. Перевод осуществляется через гиперболическую аномалию.

    :param true_anomaly: вектор истинных аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор средних аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    anomaly = true_anomaly_to_hyperbolic_anomaly(true_anomaly, eccentricity)
    mean_anomaly = hyperbolic_anomaly_to_mean_anomaly(anomaly, eccentricity)
    return mean_anomaly


def mean_anomaly_to_true_anomaly_hyperbolic(mean_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из средней аномалии в истинную. Перевод осуществляется через гиперболическую аномалию.

    :param mean_anomaly: вектор средних аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор истинных аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    hyperbolic_anomaly = mean_anomaly_to_hyperbolic_anomaly(mean_anomaly, eccentricity)
    mean_anomaly = hyperbolic_anomaly_to_true_anomaly(hyperbolic_anomaly, eccentricity)
    return mean_anomaly


def hyperbolic_anomaly_to_mean_anomaly(hyperbolic_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из гиперболической аномалии в среднюю.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a, p. 67 (3.138)
    :param eccentric_anomaly: вектор гиперболических аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор средних аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    return eccentricity * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly


def mean_anomaly_to_hyperbolic_anomaly(mean_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из средней аномалии в гиперболическую.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a, p. 76 section 3.3.1
    :param mean_anomaly: вектор средних аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор гиперболических аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    eccentricity_small = eccentricity < 1.6
    eccentricity_not_small = np.logical_not(eccentricity_small)

    mean_anomaly_diff_condition = np.any([mean_anomaly > np.pi,
                                          np.all([mean_anomaly < 0, mean_anomaly > - np.pi], axis=0)], axis=0)

    eccentricity_mean = np.all([np.abs(mean_anomaly) > np.pi, eccentricity < 3.6], axis=0)

    # pylint: disable=invalid-name
    H_res = np.copy(mean_anomaly)
    first_condition = np.all([eccentricity_small, mean_anomaly_diff_condition], axis=0)
    second_condition = np.all([eccentricity_small, np.logical_not(mean_anomaly_diff_condition)], axis=0)
    third_condition = np.all([eccentricity_not_small, eccentricity_mean], axis=0)
    four_condition = np.all([eccentricity_not_small, np.logical_not(eccentricity_mean)], axis=0)

    H_res[first_condition] = mean_anomaly[first_condition] - eccentricity[first_condition]
    H_res[second_condition] = mean_anomaly[second_condition] + eccentricity[second_condition]
    H_res[third_condition] = (mean_anomaly[third_condition]
                              - eccentricity[third_condition] * np.sign(mean_anomaly[third_condition]))
    H_res[four_condition] = mean_anomaly[four_condition] / (eccentricity[four_condition] - 1)

    # pylint: disable=invalid-name
    H_res = np.where(np.abs(H_res) > 10, np.arcsinh(H_res), H_res)

    index = H_res > H_res - 1
    while np.any(index):
        H = H_res[index]  # pylint: disable=invalid-name
        # pylint: disable=invalid-name
        H_next = H + ((mean_anomaly[index] + H - eccentricity[index] * np.sinh(H))
                      / (eccentricity[index] * np.cosh(H) - 1))
        need_computation = np.abs(H_next - H) > 1e-8
        H_res[index] = H_next
        index[index] = need_computation
    return H_res


def true_anomaly_to_hyperbolic_anomaly(true_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из средней аномалии в гиперболическую.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a c. 64
    :param true_anomaly: вектор истинных аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор гиперболических аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    hyperbolic_pre_sin = np.sin(true_anomaly) * np.sqrt(eccentricity * eccentricity - 1)
    divisor = 1 + eccentricity * np.cos(true_anomaly)
    return np.arcsinh(hyperbolic_pre_sin / divisor)


def hyperbolic_anomaly_to_true_anomaly(hyperbolic_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Перевод из гиперболической аномалии в истинную.

    Вычисления проводятся согласно GMAT Mathematical Specifications R2018a p.77 (3.213)
    :param hyperbolic_anomaly: вектор гмперболических аномалий. Может быть скаляром.
    :param eccentricity: вектор эксцентриситетов. Может быть скаляром.
    :return: вектор истинных аномалий, такого же размера как и входные данные.
             Если оба входных параметра скаляры, то результат будет скаляром
    '''
    true_pre_sin = -np.sinh(hyperbolic_anomaly) * np.sqrt(eccentricity * eccentricity - 1)
    true_pre_cos = np.cosh(hyperbolic_anomaly) - eccentricity
    divisor = 1 - eccentricity * np.cosh(hyperbolic_anomaly)
    return np.arctan2(true_pre_sin / divisor, true_pre_cos / divisor)


def keplerian_to_cartesian(semilatus: ArrayFloat, eccentricity: ArrayFloat,
                           inclination: ArrayFloat, argument_periapsis: ArrayFloat, ascending_node: ArrayFloat,
                           true_anomaly: ArrayFloat, gravitational_parameter: float) -> List[ArrayVector3]:
    '''
    Функция для перевода Кепплеровых элементов орбиты в декартовы координаты и скорость

    :param semimajor: Большая полуось. Вектор из N чисел.
    :param eccentricity: Эксцентриситет. Вектор из N чисел.
    :param inclination: Наклонение. Вектор из N чисел.
    :param argument_periapsis: Аргумент перицентра. Вектор из N чисел.
    :param ascending_node: Долгота восходящего узла. Вектор из N чисел.
    :param true_anomaly: Истинная аномалия. Вектор из N чисел.

    .. note:: Для случаев, когда исходные данные выходят за пределы области определения,
            то есть :math:`1 + cos(\\nu) <= 0`, то
    '''

    cos_true_anomaly = np.cos(true_anomaly)
    domain_of_function = 1 + eccentricity * cos_true_anomaly > 1e-11

    position = np.zeros((len(semilatus), 3)) + np.nan
    velocity = np.zeros((len(semilatus), 3)) + np.nan

    cos_inclination = np.cos(inclination[domain_of_function])
    sin_inclination = np.sin(inclination[domain_of_function])

    sin_periapsis = np.sin(argument_periapsis[domain_of_function])
    cos_periapsis = np.cos(argument_periapsis[domain_of_function])

    cos_ascending = np.cos(ascending_node[domain_of_function])
    sin_ascending = np.sin(ascending_node[domain_of_function])

    cos_true_anomaly = np.cos(true_anomaly[domain_of_function])
    sin_true_anomaly = np.sin(true_anomaly[domain_of_function])

    rot_11 = cos_ascending * cos_periapsis - sin_ascending * sin_periapsis * cos_inclination
    rot_21 = sin_ascending * cos_periapsis + cos_ascending * sin_periapsis * cos_inclination
    rot_31 = sin_periapsis * sin_inclination

    rot_12 = - cos_ascending * sin_periapsis - sin_ascending * cos_periapsis * cos_inclination
    rot_22 = - sin_ascending * sin_periapsis + cos_ascending * cos_periapsis * cos_inclination
    rot_32 = cos_periapsis * sin_inclination

    # rot_13 = sin_ascending * sin_inclination
    # rot_23 = - cos_ascending * sin_inclination
    # rot_33 = cos_inclination

    distance = semilatus[domain_of_function] / (1 + eccentricity[domain_of_function] * cos_true_anomaly)
    pos_1 = distance * cos_true_anomaly
    pos_2 = distance * sin_true_anomaly
    pos_3 = 0  # Подчеркиваем, что величина равна 0 pylint: disable=unused-variable  # noqa: F841

    speed = np.sqrt(gravitational_parameter / semilatus[domain_of_function])
    vel_1 = -speed * sin_true_anomaly
    vel_2 = speed * (eccentricity[domain_of_function] + cos_true_anomaly)
    vel_3 = 0  # Подчеркиваем, что величина равна 0 pylint: disable=unused-variable  # noqa: F841

    position[domain_of_function] = np.column_stack((rot_11 * pos_1 + rot_12 * pos_2,
                                                    rot_21 * pos_1 + rot_22 * pos_2,
                                                    rot_31 * pos_1 + rot_32 * pos_2))
    velocity[domain_of_function] = np.column_stack((rot_11 * vel_1 + rot_12 * vel_2,
                                                    rot_21 * vel_1 + rot_22 * vel_2,
                                                    rot_31 * vel_1 + rot_32 * vel_2))

    return position, velocity


def translate_anomaly(src_anomaly: ArrayFloat, eccentricity: ArrayFloat, func_eccentric, func_hyperbolic) -> ArrayFloat:
    '''
    Преобразование между 2 аномалиями(величинами) на основе типа орбиты(эллиптическая или гиперболическая).

    :param src_anomaly: вектор исходных величин
    :param eccentricity: вектор эксцентриситетов. На основе него определяется какую функцию использовать для расчетов
    :param func_eccentric: функция применяемая к эллиптическим орбитам: эксцентриситет меньше-либо равен 1
    :param func_hyperbolic: функция применяемая к гиперболическим орбитам: эксцентриситет больше 1
    '''
    res = np.copy(src_anomaly)
    elliptic_orbit = np.all(np.array([eccentricity < 1, np.logical_not(np.isnan(src_anomaly))]), axis=0)
    # print('aaaa', np.array([eccentricity > 1, not np.isnan(src_anomaly)]))
    hyperbolic_orbit = np.all(np.array([eccentricity > 1, np.logical_not(np.isnan(src_anomaly))]), axis=0)
    if np.any(elliptic_orbit):
        func_indexed_params_save_res(res, func_eccentric, elliptic_orbit, src_anomaly, eccentricity)
    if np.any(hyperbolic_orbit):
        func_indexed_params_save_res(res, func_hyperbolic, hyperbolic_orbit, src_anomaly, eccentricity)
    return res


def mean_anomaly_to_true_anomaly(mean_anomaly: ArrayFloat, eccentricity: ArrayFloat) -> ArrayFloat:
    '''
    Преобразование средней аномалии в истинную.

    По состоянию на 31.01.2020 не реализовано преобразование для параболических траекторий.
    Для них результат возвращается без изменений.

    :param src_anomaly: вектор истинных аномалий
    :param eccentricity: вектор эксцентриситетов

    :return: вектор средних аномалий
    :rtype: ndarray(float)
    '''
    return translate_anomaly(mean_anomaly, eccentricity,
                             mean_anomaly_to_true_anomaly_eccentric,
                             mean_anomaly_to_true_anomaly_hyperbolic)
