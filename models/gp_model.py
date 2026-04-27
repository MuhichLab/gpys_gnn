"""Simple Gaussian Process model for atomistic structure regression."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class AtomicGaussianProcess:
    """Gaussian Process with SOAP-like atomic descriptors.

    Each structure is represented by a matrix of atomic descriptors with shape
    ``(n_atoms, n_features)``. Structure-level kernels are computed as the sum
    over all atomic pair kernels.
    """

    def __init__(self, lengthscale: float = 1.0, noise: float = 1e-5) -> None:
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)

        self.X_list: list[np.ndarray] | None = None
        self.y: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    def atomic_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute vectorized RBF kernel between two atomic descriptor sets.

        Parameters
        ----------
        x1
            Descriptor matrix with shape ``(n1, d)``.
        x2
            Descriptor matrix with shape ``(n2, d)``.

        Returns
        -------
        np.ndarray
            Kernel matrix with shape ``(n1, n2)``.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        sq_norm_1 = np.sum(x1 * x1, axis=1, keepdims=True)
        sq_norm_2 = np.sum(x2 * x2, axis=1, keepdims=True).T
        sq_dist = sq_norm_1 + sq_norm_2 - 2.0 * (x1 @ x2.T)
        sq_dist = np.maximum(sq_dist, 0.0)

        denom = 2.0 * (self.lengthscale**2)
        return np.exp(-sq_dist / denom)

    def structure_kernel(self, A: np.ndarray, B: np.ndarray) -> float:
        """Compute structure-level kernel as sum over atomic pair kernels."""
        k_atoms = self.atomic_kernel(A, B)
        return float(np.sum(k_atoms))

    def build_kernel_matrix(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Build symmetric kernel matrix for a sequence of structures."""
        n_structures = len(X_list)
        K = np.zeros((n_structures, n_structures), dtype=float)

        for i in range(n_structures):
            K[i, i] = self.structure_kernel(X_list[i], X_list[i])
            for j in range(i + 1, n_structures):
                kij = self.structure_kernel(X_list[i], X_list[j])
                K[i, j] = kij
                K[j, i] = kij

        return K

    def fit(self, X_list: Sequence[np.ndarray], y: Iterable[float]) -> None:
        """Fit GP weights for training structures and targets."""
        X_train = [np.asarray(X, dtype=float) for X in X_list]
        y_train = np.asarray(list(y), dtype=float)

        if len(X_train) != len(y_train):
            raise ValueError("X_list and y must have the same number of samples.")

        K = self.build_kernel_matrix(X_train)
        K = K + self.noise * np.eye(K.shape[0], dtype=float)

        jitter = 1e-10
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K = K + jitter * np.eye(K.shape[0], dtype=float)
            L = np.linalg.cholesky(K)


        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train) + 1e-12

        y_scaled = (y_train - self.y_mean) / self.y_std

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))

        self.X_list = X_train
        self.y = y_train
        self.K = K
        self.L = L
        self.alpha = alpha


    def _compute_k_star(self, X: np.ndarray) -> np.ndarray:
        """Kernel vector between a test structure and all training structures."""
        if self.X_list is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return np.array(
            [self.structure_kernel(X, X_train) for X_train in self.X_list],
            dtype=float,
        )

    def _compute_k_self(self, X: np.ndarray) -> float:
        """Kernel value of a structure with itself."""
        return self.structure_kernel(X, X)

    def predict_single(
        self,
        X: np.ndarray,
        return_mean: bool = True,
        return_std: bool = True,
    ) -> tuple[float | None, float | None]:
        """Predict mean/std for one structure with shared kernel computations."""
        if self.alpha is None or self.L is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=float)
        k_star = self._compute_k_star(X)

        mean = float(k_star @ self.alpha)*self.y_std + self.y_mean if return_mean else None
        k_self = self._compute_k_self(X) if return_std else None

        std = None
        if return_std:
            v = np.linalg.solve(self.L, k_star)
            var =k_self - v @ v
            var = max(var,1e-12)
            std = float(np.sqrt(var))*self.y_std 

        return mean, std

    def predict(self, X_list: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation for test structures."""
        X_test = [np.asarray(X, dtype=float) for X in X_list]
        mu = np.zeros(len(X_test), dtype=float)
        sigma = np.zeros(len(X_test), dtype=float)

        for idx, X in enumerate(X_test):
            mean, std = self.predict_single(X, return_mean=True, return_std=True)
            mu[idx] = 0.0 if mean is None else mean
            sigma[idx] = 0.0 if std is None else std

        return mu, sigma

    def predict_mean(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Predict only GP mean for test structures."""
        X_test = [np.asarray(X, dtype=float) for X in X_list]
        mu = np.zeros(len(X_test), dtype=float)

        for idx, X in enumerate(X_test):
            mean, _ = self.predict_single(X, return_mean=True, return_std=False)
            mu[idx] = 0.0 if mean is None else mean

        return mu

    def predict_uncertainty(self, X_list: Sequence[np.ndarray]) -> np.ndarray:
        """Predict only GP standard deviation for test structures."""
        X_test = [np.asarray(X, dtype=float) for X in X_list]
        sigma = np.zeros(len(X_test), dtype=float)

        for idx, X in enumerate(X_test):
            _, std = self.predict_single(X, return_mean=False, return_std=True)
            sigma[idx] = 0.0 if std is None else std

        return sigma

