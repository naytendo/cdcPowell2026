# models/learned_model.py

import numpy as np
from .discrete_model import DiscreteModel

class ResidualAugmentedModel(DiscreteModel):
    """
    A discrete-time model defined as:
        f_new(x,u) = f_nom(x,u) + residual_model.predict(x,u)
    """

    def __init__(self, model_nom, residual_model):
        super().__init__(cont_sys= model_nom.cont_sys, t=model_nom.t)
        self.model_nom = model_nom
        self.residual_model = residual_model

    def step(self, x, u, k):
        x_nom = self.model_nom.step(x, u, k)
        e_hat = self.residual_model.predict(x, u)
        return x_nom + e_hat

    def linearize(self, traj):
        # Option 1: use nominal jacobians only (common in practice)
        return self.model_nom.linearize(traj)

        # Option 2: if residual_model has jacobians, combine them
        # but most learned models won't have analytic jacobians
