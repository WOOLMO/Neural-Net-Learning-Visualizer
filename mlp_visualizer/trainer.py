from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .datasets import make_dataset
from .network import MLP, LayerGrads, binary_cross_entropy


@dataclass
class TrainingState:
    dataset_name: str = "moons"
    hidden_layers: tuple[int, ...] = (6, 4)
    learning_rate: float = 0.08
    seed: int = 7
    step: int = 0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x, self.y = make_dataset(self.dataset_name, seed=self.seed)
        self.net = MLP([2, *self.hidden_layers, 1], seed=self.seed)
        self.last_grads = [LayerGrads(np.zeros_like(w), np.zeros_like(b)) for w, b in zip(self.net.weights, self.net.biases)]

    def reset(self, dataset: str, hidden_layers: list[int], learning_rate: float, seed: int) -> None:
        self.dataset_name = dataset
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.seed = seed
        self.step = 0
        self.history.clear()
        self.__post_init__()

    def train(self, steps: int) -> None:
        for _ in range(max(1, min(steps, 200))):
            result = self.net.train_step(self.x, self.y, self.learning_rate)
            self.last_grads = result["grads"]
            self.step += 1
            self.history.append({
                "step": self.step,
                "loss": result["loss"],
                "accuracy": result["accuracy"],
            })
            self.history = self.history[-400:]

    def snapshot(self) -> dict:
        probabilities = self.net.predict_proba(self.x)
        loss = binary_cross_entropy(self.y, probabilities)
        accuracy = float(np.mean((probabilities >= 0.5) == self.y))
        grid = self._decision_grid()
        sample_activations = self._sample_activations()

        return {
            "step": self.step,
            "loss": loss,
            "accuracy": accuracy,
            "datasetName": self.dataset_name,
            "dataset": {
                "x": self.x.round(4).tolist(),
                "y": self.y.flatten().astype(int).tolist(),
            },
            "network": {
                "layerSizes": self.net.layer_sizes,
                "weights": [w.round(4).tolist() for w in self.net.weights],
                "biases": [b.flatten().round(4).tolist() for b in self.net.biases],
                "gradWeights": [g.d_w.round(4).tolist() for g in self.last_grads],
                "gradBiases": [g.d_b.flatten().round(4).tolist() for g in self.last_grads],
                "sampleActivations": sample_activations,
            },
            "history": self.history,
            "grid": grid,
        }

    def _decision_grid(self, resolution: int = 42) -> dict:
        xs = np.linspace(-2.6, 2.6, resolution)
        ys = np.linspace(-2.6, 2.6, resolution)
        xx, yy = np.meshgrid(xs, ys)
        points = np.c_[xx.ravel(), yy.ravel()]
        zz = self.net.predict_proba(points).reshape(resolution, resolution)
        return {
            "xs": xs.round(4).tolist(),
            "ys": ys.round(4).tolist(),
            "z": zz.round(4).tolist(),
        }

    def _sample_activations(self) -> list[list[float]]:
        probe = np.array([[0.9, 0.65]], dtype=np.float64)
        activations, _ = self.net.forward(probe)
        return [layer.flatten().round(4).tolist() for layer in activations]
