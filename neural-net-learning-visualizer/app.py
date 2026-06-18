from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from mlp_visualizer.trainer import TrainingState


app = Flask(__name__)
state = TrainingState()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def get_state():
    return jsonify(state.snapshot())


@app.post("/api/train")
def train():
    payload = request.get_json(silent=True) or {}
    state.train(int(payload.get("steps", 1)))
    return jsonify(state.snapshot())


@app.post("/api/reset")
def reset():
    payload = request.get_json(silent=True) or {}
    hidden_layers = payload.get("hiddenLayers", [6, 4])
    hidden_layers = [max(1, min(10, int(size))) for size in hidden_layers][:3]
    state.reset(
        dataset=payload.get("dataset", "moons"),
        hidden_layers=hidden_layers,
        learning_rate=float(payload.get("learningRate", 0.08)),
        seed=int(payload.get("seed", 7)),
    )
    return jsonify(state.snapshot())


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
