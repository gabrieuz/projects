"""
Funções de custo e cálculo de gradientes para regressão linear.
"""
import numpy as np


def compute_cost(x, y, w, b):
    """
    Calcula a função de custo MSE (Mean Squared Error) para regressão linear.

    Args:
        x (ndarray): dados de entrada, formato (m,)
        y (ndarray): alvos, formato (m,)
        w (float): parâmetro w do modelo
        b (float): parâmetro b do modelo

    Returns:
        float: valor da função de custo
    """
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


def compute_gradient(x, y, w, b):
    """
    Calcula o gradiente da função de custo MSE em relação aos parâmetros w e b.

    Args:
        x (ndarray): dados de entrada, formato (m,)
        y (ndarray): alvos, formato (m,)
        w (float): parâmetro w do modelo
        b (float): parâmetro b do modelo

    Returns:
        tuple: gradientes dj_dw, dj_db
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def compute_model_output(x, w, b):
    """
    Calcula a saída do modelo de regressão linear para um conjunto de entradas.

    Args:
        x (ndarray): dados de entrada, formato (m,)
        w (float): parâmetro w do modelo
        b (float): parâmetro b do modelo

    Returns:
        ndarray: predições do modelo, formato (m,)
    """
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb
