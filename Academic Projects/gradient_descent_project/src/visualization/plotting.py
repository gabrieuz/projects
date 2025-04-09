"""
Funções para visualização e comparação dos algoritmos de gradient descent.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Cores para visualização
dlblue = '#0096ff'
dlorange = '#FF9300'
dldarkred = '#C00000'
dlmagenta = '#FF40FF'
dlpurple = '#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]


def plot_data_and_model(x, y, w, b, title="Dados e Modelo"):
    """
    Plota os dados e a linha do modelo.

    Args:
        x (ndarray): dados de entrada
        y (ndarray): alvos
        w (float): parâmetro w do modelo (inclinação)
        b (float): parâmetro b do modelo (intercepto)
        title (str): título do gráfico
    """
    # Ordena os dados para visualização adequada da linha
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Calcula as predições do modelo
    y_pred = w * x_sorted + b

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker='x', c='r', label='Dados Reais')
    plt.plot(x_sorted, y_pred, c='b', label='Predição do Modelo')
    plt.title(title)
    plt.xlabel("ft² (em milhares)")
    plt.ylabel("Preço (em milhares)")
    plt.legend()
    plt.grid(True)
    plt.show()


def inbounds(p, base, xlim, ylim):
    """
    Verifica se os pontos p e base estão dentro dos limites do gráfico.

    Args:
        p (list): ponto atual
        base (list): ponto base
        xlim (tuple): limites do eixo x
        ylim (tuple): limites do eixo y

    Returns:
        bool: True se ambos os pontos estão dentro dos limites
    """
    return (xlim[0] <= p[0] <= xlim[1]) and (ylim[0] <= p[1] <= ylim[1]) and \
           (xlim[0] <= base[0] <= xlim[1]) and (ylim[0] <= base[1] <= ylim[1])


def plot_contour_with_path(x, y, w_history, b_history, cost_function, title="Contour Plot"):
    """
    Plota o contorno da função de custo e o caminho percorrido pelo algoritmo.

    Args:
        x (ndarray): dados de entrada
        y (ndarray): alvos
        w_history (list): histórico de valores de w
        b_history (list): histórico de valores de b
        cost_function (function): função de custo
        title (str): título do gráfico
    """
    # Cria grid para o contorno
    w_range = np.linspace(min(w_history) - 20, max(w_history) + 20, 100)
    b_range = np.linspace(min(b_history) - 20, max(b_history) + 20, 100)

    W, B = np.meshgrid(w_range, b_range)
    Z = np.zeros_like(W)

    # Calcula o valor da função de custo para cada ponto do grid
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = cost_function(x, y, W[i, j], B[i, j])

    # Plota o contorno
    plt.figure(figsize=(12, 8))
    contours = plt.contour(W, B, Z, levels=[10, 50, 100, 500, 1000, 5000, 10000],
                           colors=dlcolors, linewidths=2)
    plt.clabel(contours, inline=1, fontsize=10)

    # Plota o caminho
    plt.plot(w_history, b_history, '-o', color='black', markersize=4)

    # Destaca o ponto inicial e final
    plt.plot(w_history[0], b_history[0], 'ro',
             markersize=10, label='Ponto Inicial')
    plt.plot(w_history[-1], b_history[-1], 'go',
             markersize=10, label='Ponto Final')

    plt.title(title)
    plt.xlabel('w (peso)')
    plt.ylabel('b (viés)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plt_contour_wgrad(x, y, hist, ax, title, w_range=[-100, 500, 5], b_range=[-100, 500, 5],
                      contours=[1, 10, 50, 100, 500, 1000, 5000, 10000], resolution=5, step=10):
    """
    Plota o contorno da função de custo e o caminho percorrido pelo algoritmo com setas.

    Args:
        x (ndarray): dados de entrada
        y (ndarray): alvos
        hist (list): histórico de parâmetros [w, b] em cada iteração
        ax (matplotlib.axes): eixo para o plot
        title (str): título do gráfico
        w_range (list): [wmin, wmax, passos] para o eixo w
        b_range (list): [bmin, bmax, passos] para o eixo b
        contours (list): níveis de contorno para desenhar
        resolution (float): distância mínima entre pontos para desenhar seta
        step (int): salto entre pontos para reduzir o número de setas
    """
    b0, w0 = np.meshgrid(np.arange(*b_range), np.arange(*w_range))
    z = np.zeros_like(b0)
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            z[i][j] = compute_cost(x, y, w0[i][j], b0[i][j])

    CS = ax.contour(w0, b0, z, contours, linewidths=2,
                    colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple])
    ax.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title(title)

    # Destaca o ponto final
    w_final = hist[-1][0]
    b_final = hist[-1][1]
    ax.hlines(b_final, ax.get_xlim()[0], w_final,
              lw=2, color=dlpurple, ls='dotted')
    ax.vlines(w_final, ax.get_ylim()[0], b_final,
              lw=2, color=dlpurple, ls='dotted')

    # Desenha o caminho com setas
    base = hist[0]
    for point in hist[0::step]:
        edist = np.sqrt((base[0] - point[0]) ** 2 + (base[1] - point[1]) ** 2)
        if edist > resolution or point == hist[-1]:
            if inbounds(point, base, ax.get_xlim(), ax.get_ylim()):
                plt.annotate('', xy=point, xytext=base, xycoords='data',
                             arrowprops={'arrowstyle': '->',
                                         'color': 'r', 'lw': 1},
                             va='center', ha='center')
            base = point


def compare_cost_histories(J_histories, labels, title="Comparação do Histórico de Custo"):
    """
    Compara os históricos de custo de diferentes algoritmos.

    Args:
        J_histories (list): lista de históricos de custo
        labels (list): lista de nomes dos algoritmos
        title (str): título do gráfico
    """
    plt.figure(figsize=(12, 6))

    for i, history in enumerate(J_histories):
        plt.plot(history, label=labels[i])

    plt.title(title)
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_contour_plots(x, y, histories, labels, cost_function, title="Comparação de Contornos"):
    """
    Compara os caminhos de diferentes algoritmos em gráficos de contorno.

    Args:
        x (ndarray): dados de entrada
        y (ndarray): alvos
        histories (list): lista de históricos de parâmetros [w, b]
        labels (list): lista de nomes dos algoritmos
        cost_function (function): função de custo
        title (str): título do gráfico
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(5*len(histories), 6))

    for i, (hist, label) in enumerate(zip(histories, labels)):
        plt_contour_wgrad(x, y, hist, axes[i], label,
                          w_range=[-100, 400, 5], b_range=[-50, 200, 5],
                          contours=[265, 600, 1500, 5000, 10000], resolution=10)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Função temporária para compatibilidade
def compute_cost(x, y, w, b):
    """
    Função de custo temporária para compatibilidade interna do módulo.
    Na prática, esta função seria importada do módulo cost_functions.
    """
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost
