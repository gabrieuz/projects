"""
Módulo de otimizadores para gradiente descendente.
"""
from .gradient_descent import batch_gradient_descent, stochastic_gradient_descent, minibatch_gradient_descent

__all__ = [
    'batch_gradient_descent',
    'stochastic_gradient_descent',
    'minibatch_gradient_descent'
]
