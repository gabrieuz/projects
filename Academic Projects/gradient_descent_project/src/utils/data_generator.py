"""
Módulo para geração de dados sintéticos para testes dos algoritmos de gradient descent.
"""
import numpy as np


def generate_house_prices_data(m=100, seed=42):
    """
    Gera dados sintéticos simulando preços de casas baseados em tamanho.

    Args:
        m (int): número de amostras para gerar
        seed (int): semente para reproduzibilidade dos resultados

    Returns:
        tuple: x_train (tamanhos), y_train (preços), w_real, b_real (parâmetros reais)
    """
    # Define os parâmetros reais do modelo
    w_real = 120  # Inclinação: cada mil pés² adiciona $120K
    b_real = 80   # Intercepto: preço base de $80K

    # Gera tamanhos de casas aleatórios entre 0 e 4 mil pés²
    np.random.seed(seed)
    x_train = np.random.uniform(low=0.0, high=4.0, size=m)

    # Calcula os preços exatos com base nos parâmetros reais
    y_exact = w_real * x_train + b_real

    # Adiciona ruído gaussiano com desvio padrão 20
    noise = np.random.normal(0, 20, size=m)
    y_train = y_exact + noise

    return x_train, y_train, w_real, b_real


def split_data(x, y, train_ratio=0.8, seed=None):
    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        x (ndarray): dados de entrada
        y (ndarray): alvos
        train_ratio (float): proporção de dados para treino (0-1)
        seed (int): semente para reproduzibilidade

    Returns:
        tuple: x_train, y_train, x_test, y_test
    """
    if seed is not None:
        np.random.seed(seed)

    m = x.shape[0]
    indices = np.random.permutation(m)
    train_size = int(m * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_test, y_test
