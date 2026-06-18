from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerGrads:
    d_w: np.ndarray
    d_b: np.ndarray


class MLP:
    def __init__(self, layer_sizes: list[int], seed: int = 3) -> None:
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0, scale, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        pre_activations = []

        for index, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            pre_activations.append(z)
            if index == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = np.tanh(z)
            activations.append(a)

        return activations, pre_activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0][-1]

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> dict:
        activations, pre_activations = self.forward(x)
        y_hat = activations[-1]
        loss = binary_cross_entropy(y, y_hat)
        accuracy = float(np.mean((y_hat >= 0.5) == y))

        delta = (y_hat - y) / len(x)
        grads: list[LayerGrads] = []

        for layer_index in reversed(range(len(self.weights))):
            d_w = activations[layer_index].T @ delta
            d_b = np.sum(delta, axis=0, keepdims=True)
            grads.append(LayerGrads(d_w=d_w, d_b=d_b))

            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * tanh_derivative(pre_activations[layer_index - 1])

        grads.reverse()

        for index, grad in enumerate(grads):
            self.weights[index] -= learning_rate * grad.d_w
            self.biases[index] -= learning_rate * grad.d_b

        return {
            "loss": float(loss),
            "accuracy": accuracy,
            "grads": grads,
            "activations": activations,
        }


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    a = np.tanh(z)
    return 1 - a * a


def binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    eps = 1e-8
    return float(-np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))
