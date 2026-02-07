# learning/residual_model.py

from abc import ABC, abstractmethod
import numpy as np

class ResidualModel(ABC):
    """
    Abstract base class for residual dynamics models.
    
    A residual model learns the mapping:
        (x, u) -> e
    where e is the one-step prediction error:
        e = f_true(x, u) - f_nom(x, u)
    """

    @abstractmethod
    def fit(self, X, U, E):
        """
        Fit the residual model using training data.

        Parameters
        ----------
        X : array, shape (N, nx)
            State samples.
        U : array, shape (N, nu)
            Input samples.
        E : array, shape (N, nx)
            Residual samples.
        """
        pass

    @abstractmethod
    def predict(self, x, u):
        """
        Predict the residual e_hat for a single (x, u) pair.

        Parameters
        ----------
        x : array, shape (nx,)
        u : array, shape (nu,)

        Returns
        -------
        e_hat : array, shape (nx,)
        """
        pass

    def predict_batch(self, X, U):
        """
        Optional convenience method for batch prediction.

        Parameters
        ----------
        X : array, shape (N, nx)
        U : array, shape (N, nu)

        Returns
        -------
        E_hat : array, shape (N, nx)
        """
        return np.array([self.predict(x, u) for x, u in zip(X, U)])
