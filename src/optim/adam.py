"""
Adam optimizer (adaptive moment estimation).

Adam keeps two per-parameter running averages: a first moment `m`
(an exponential moving average of the gradients) and a second
moment `v` (an EMA of the squared gradients). Each step uses
bias-corrected versions of these moments to scale the update, so
parameters with consistently small gradients still take meaningful
steps while noisy directions are damped.

The update for each parameter, at step `t`:

    m = β1 * m + (1 - β1) * grad
    v = β2 * v + (1 - β2) * grad²
    m_hat = m / (1 - β1ᵗ)
    v_hat = v / (1 - β2ᵗ)
    param -= lr * m_hat / (sqrt(v_hat) + eps)
"""

from collections.abc import Iterable

import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer, mirroring `torch.optim.Adam`.

    Constructor arguments:
        parameters -- iterable of `(parameter, gradient)` array pairs,
                      typically `model.parameters()`. As with `SGD`,
                      `backward` must mutate both arrays in place so the
                      optimizer keeps stable references.
        lr         -- learning rate
        betas      -- (β1, β2): decay rates for the first- and
                      second-moment estimates.
        eps        -- term added to the denominator for numerical
                      stability.
        weight_decay -- L2 penalty coefficient. When non-zero,
                      `weight_decay * param` is added to the gradient
                      before the moment updates (coupled L2, matching
                      `torch.optim.Adam`; this is not the decoupled
                      AdamW variant).
    """

    def __init__(
        self,
        parameters: Iterable[tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m: list[np.ndarray] = [
            np.zeros_like(param) for param, _ in self.parameters
        ]
        self._v: list[np.ndarray] = [
            np.zeros_like(param) for param, _ in self.parameters
        ]

    def step(self) -> None:
        """Applies one update step to every parameter."""
        self.t += 1
        m_hat_denom = 1 - self.beta1 ** self.t
        v_hat_denom = 1 - self.beta2 ** self.t
        for i, (param, grad) in enumerate(self.parameters):
            m = self._m[i]
            v = self._v[i]
            
            _grad = grad
            if self.weight_decay != 0:
                _grad = grad + self.weight_decay * param
                
            m *= self.beta1
            m += (1 - self.beta1) * _grad
            v *= self.beta2
            v += (1 - self.beta2) * _grad**2
            
            m_hat = m / m_hat_denom
            v_hat = v / v_hat_denom
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
