# final_project/src/model_mlp.py
from __future__ import annotations
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


class MLP:
    """
    2-layer MLP: input -> ReLU(hidden) -> softmax(output)
    Pure NumPy, real backprop.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim), dtype=np.float64)

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        p = softmax(z2)
        cache = (X, z1, a1, z2, p)
        return p, cache

    def loss(self, y: np.ndarray, p: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(np.log(p[np.arange(len(y)), y])))

    def backward(self, cache, y: np.ndarray):
        X, z1, a1, z2, p = cache
        n = X.shape[0]

        yoh = one_hot(y, p.shape[1])
        dz2 = (p - yoh) / n

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def step(self, grads, lr: float):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X: np.ndarray):
        p, _ = self.forward(X)
        return np.argmax(p, axis=1), p

    def save(self, path: str, meta: dict | None = None):
        if meta is None:
            meta = {}
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            **{f"meta_{k}": np.array([v], dtype=object) for k, v in meta.items()}
        )

    @staticmethod
    def load(path: str):
        data = np.load(path, allow_pickle=True)
        W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
        model = MLP(W1.shape[0], W1.shape[1], W2.shape[1])
        model.W1, model.b1, model.W2, model.b2 = W1, b1, W2, b2
        return model
