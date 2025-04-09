# Implementações de Gradient Descent

Este projeto demonstra três implementações diferentes do algoritmo de gradient descent para regressão linear:

1. **Batch Gradient Descent** - Utiliza o conjunto completo de dados para calcular o gradiente a cada iteração
2. **Stochastic Gradient Descent (SGD)** - Utiliza apenas uma única instância aleatória para calcular o gradiente
3. **Mini-batch Gradient Descent** - Utiliza um pequeno conjunto aleatório de instâncias para calcular o gradiente

## Estrutura do Projeto

```
gradient_descent_project/
├── src/
│   ├── optimizers/ - Implementações dos algoritmos de gradient descent
│   ├── utils/ - Funções auxiliares (cálculo de custo, gradientes, geração de dados)
│   └── visualization/ - Código para visualização e comparação dos algoritmos
├── notebooks/ - Notebooks Jupyter para demonstrações interativas
├── data/ - Dados gerados ou utilizados pelo projeto
└── tests/ - Testes unitários
```

## Como Usar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute o script principal:

```bash
python main.py
```

3. Ou explore os notebooks interativos:

```bash
jupyter notebook notebooks/gradient_descent_comparison.ipynb
```

## Referências Teóricas

-   Gradient Descent é um algoritmo de otimização para encontrar o mínimo de uma função
-   Batch GD usa todo o dataset para cada atualização de parâmetros
-   SGD usa apenas uma instância aleatória para cada atualização
-   Mini-batch GD usa um pequeno conjunto aleatório de instâncias, combinando as vantagens das duas abordagens anteriores
