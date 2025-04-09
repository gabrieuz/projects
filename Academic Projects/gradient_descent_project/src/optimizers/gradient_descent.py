"""
Implementações de diferentes variantes do algoritmo de gradient descent.
"""
import numpy as np
import copy
import math


def batch_gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Implementa o algoritmo de Batch Gradient Descent.

    Args:
        x (ndarray): dados de treinamento, formato (m,)
        y (ndarray): alvos de treinamento, formato (m,)
        w_in (float): valor inicial do parâmetro w
        b_in (float): valor inicial do parâmetro b
        alpha (float): taxa de aprendizagem
        num_iters (int): número de iterações para executar o gradient descent
        cost_function (function): função que calcula o custo
        gradient_function (function): função que calcula o gradiente

    Returns:
        w (float): valor otimizado do parâmetro w
        b (float): valor otimizado do parâmetro b
        J_history (list): histórico de custo em cada iteração
        p_history (list): histórico de parâmetros (w,b) em cada iteração
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    p_history = []

    for i in range(num_iters):
        # Calcula o gradiente usando o dataset completo
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Atualiza os parâmetros
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Armazena o histórico
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Exibe o progresso
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4}: Custo {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


def stochastic_gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Implementa o algoritmo de Stochastic Gradient Descent (SGD).

    Args:
        x (ndarray): dados de treinamento, formato (m,)
        y (ndarray): alvos de treinamento, formato (m,)
        w_in (float): valor inicial do parâmetro w
        b_in (float): valor inicial do parâmetro b
        alpha (float): taxa de aprendizagem
        num_iters (int): número de iterações para executar o gradient descent
        cost_function (function): função que calcula o custo
        gradient_function (function): função que calcula o gradiente

    Returns:
        w (float): valor otimizado do parâmetro w
        b (float): valor otimizado do parâmetro b
        J_history (list): histórico de custo em cada iteração
        p_history (list): histórico de parâmetros (w,b) em cada iteração
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    p_history = []

    m = x.shape[0]

    for i in range(num_iters):
        # Escolhe um índice aleatório
        random_instance = np.random.randint(m)
        x_i = x[random_instance]
        y_i = y[random_instance]

        # Calcula o gradiente para uma única instância
        dj_dw, dj_db = gradient_function(
            np.array([x_i]), np.array([y_i]), w, b)

        # Atualiza os parâmetros
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Armazena o histórico
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Exibe o progresso
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4}: Custo {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


def minibatch_gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function, batch_size):
    """
    Implementa o algoritmo de Mini-batch Gradient Descent.

    Args:
        x (ndarray): dados de treinamento, formato (m,)
        y (ndarray): alvos de treinamento, formato (m,)
        w_in (float): valor inicial do parâmetro w
        b_in (float): valor inicial do parâmetro b
        alpha (float): taxa de aprendizagem
        num_iters (int): número de iterações para executar o gradient descent
        cost_function (function): função que calcula o custo
        gradient_function (function): função que calcula o gradiente
        batch_size (int): tamanho do mini-batch

    Returns:
        w (float): valor otimizado do parâmetro w
        b (float): valor otimizado do parâmetro b
        J_history (list): histórico de custo em cada iteração
        p_history (list): histórico de parâmetros (w,b) em cada iteração
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    p_history = []

    m = x.shape[0]

    for i in range(num_iters):
        # Escolhe índices aleatórios na quantidade de batch_size
        random_i = np.random.choice(m, batch_size, replace=False)
        x_i = x[random_i]
        y_i = y[random_i]

        # Calcula o gradiente para o mini-batch
        dj_dw, dj_db = gradient_function(x_i, y_i, w, b)

        # Atualiza os parâmetros
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Armazena o histórico
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Exibe o progresso
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4}: Custo {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history
