"""
Script principal para execução e comparação dos diferentes algoritmos de gradient descent.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from src.utils.data_generator import generate_house_prices_data
from src.utils.cost_functions import compute_cost, compute_gradient, compute_model_output
from src.optimizers.gradient_descent import (
    batch_gradient_descent,
    stochastic_gradient_descent,
    minibatch_gradient_descent
)
from src.visualization.plotting import (
    plot_data_and_model,
    compare_cost_histories,
    compare_contour_plots
)


def main():
    """Função principal para execução do projeto."""
    print("=" * 80)
    print("COMPARAÇÃO DE IMPLEMENTAÇÕES DE GRADIENT DESCENT PARA REGRESSÃO LINEAR")
    print("=" * 80)

    # Hiperparâmetros
    w_init = 0
    b_init = 0
    alpha = 1.0e-2
    iterations = 10000
    minibatch_size = 10

    # Gera os dados sintéticos
    print("\n1. Gerando dados sintéticos para treinamento...")
    x_train, y_train, w_real, b_real = generate_house_prices_data(
        m=100, seed=42)
    print(f"   - Dimensão dos dados: {x_train.shape[0]} amostras")
    print(f"   - Parâmetros reais: w={w_real}, b={b_real}")

    # Executando os algoritmos
    print("\n2. Executando Batch Gradient Descent...")
    start_time = time.time()
    w_bgd, b_bgd, J_hist_bgd, p_hist_bgd = batch_gradient_descent(
        x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)
    bgd_time = time.time() - start_time
    print(f"   - Tempo de execução: {bgd_time:.2f} segundos")
    print(f"   - Parâmetros encontrados: w={w_bgd:.4f}, b={b_bgd:.4f}")
    print(
        f"   - Diferença para valores reais: w_diff={abs(w_bgd-w_real):.4f}, b_diff={abs(b_bgd-b_real):.4f}")

    print("\n3. Executando Stochastic Gradient Descent (SGD)...")
    start_time = time.time()
    w_sgd, b_sgd, J_hist_sgd, p_hist_sgd = stochastic_gradient_descent(
        x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)
    sgd_time = time.time() - start_time
    print(f"   - Tempo de execução: {sgd_time:.2f} segundos")
    print(f"   - Parâmetros encontrados: w={w_sgd:.4f}, b={b_sgd:.4f}")
    print(
        f"   - Diferença para valores reais: w_diff={abs(w_sgd-w_real):.4f}, b_diff={abs(b_sgd-b_real):.4f}")

    print("\n4. Executando Mini-batch Gradient Descent...")
    start_time = time.time()
    w_mbgd, b_mbgd, J_hist_mbgd, p_hist_mbgd = minibatch_gradient_descent(
        x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient, minibatch_size)
    mbgd_time = time.time() - start_time
    print(f"   - Tempo de execução: {mbgd_time:.2f} segundos")
    print(f"   - Parâmetros encontrados: w={w_mbgd:.4f}, b={b_mbgd:.4f}")
    print(
        f"   - Diferença para valores reais: w_diff={abs(w_mbgd-w_real):.4f}, b_diff={abs(b_mbgd-b_real):.4f}")

    # Comparação dos resultados
    print("\n5. Comparação dos resultados:")
    print("-" * 70)
    print(f"{'Método':<20} {'w':<10} {'b':<10} {'Tempo (s)':<15}")
    print("-" * 70)
    print(f"{'Real':<20} {w_real:<10.4f} {b_real:<10.4f} {'N/A':<15}")
    print(f"{'Batch GD':<20} {w_bgd:<10.4f} {b_bgd:<10.4f} {bgd_time:<15.2f}")
    print(f"{'SGD':<20} {w_sgd:<10.4f} {b_sgd:<10.4f} {sgd_time:<15.2f}")
    print(f"{'Mini-batch GD':<20} {w_mbgd:<10.4f} {b_mbgd:<10.4f} {mbgd_time:<15.2f}")
    print("-" * 70)

    # Visualizações
    print("\n6. Gerando visualizações...")

    # Plot de modelos
    y_pred_bgd = compute_model_output(x_train, w_bgd, b_bgd)
    y_pred_sgd = compute_model_output(x_train, w_sgd, b_sgd)
    y_pred_mbgd = compute_model_output(x_train, w_mbgd, b_mbgd)

    # Comparar históricos de custo
    compare_cost_histories(
        [J_hist_bgd[::50], J_hist_sgd[::50], J_hist_mbgd[::50]],
        ["Batch GD", f"SGD (batch_size=1)",
         f"Mini-batch GD (batch_size={minibatch_size})"],
        title="Evolução do Custo durante o Treinamento"
    )

    # Comparar caminhos no contorno
    compare_contour_plots(
        x_train, y_train,
        [p_hist_bgd, p_hist_sgd, p_hist_mbgd],
        ["Batch GD", "SGD", f"Mini-batch (size={minibatch_size})"],
        compute_cost,
        title="Comparação dos Caminhos de Otimização"
    )

    print("\nExecução concluída!")


if __name__ == "__main__":
    main()
