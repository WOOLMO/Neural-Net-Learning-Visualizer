# Neural Network Learning Visualizer

An interactive Flask and NumPy application that shows how a small multilayer perceptron learns in real time. The app trains a real neural network on 2D classification datasets and visualizes the decision boundary, weights, biases, gradients, activations, loss, and accuracy as training progresses.

## Features

- Real multilayer perceptron implemented with NumPy
- Interactive training controls for dataset, architecture, learning rate, seed, stepping, running, and resetting
- Live decision-boundary visualization for two moons, XOR, and spiral datasets
- Network graph view showing activations and weight strength
- Loss and accuracy history chart
- Parameter panel for weights, biases, and gradients

## What the Model Trains On

The model trains on synthetic 2D binary-classification datasets. Each example is a coordinate pair, `[x1, x2]`, with a target label of `0` or `1`.

- **Two moons**: two interleaving half-moon shapes that require a curved boundary.
- **XOR**: points in opposite diagonal quadrants share the same class, so a straight line is not enough.
- **Spiral**: two winding arms that make the classification task more nonlinear.

Each dataset is normalized before training. Hidden layers use `tanh` activations, the output layer uses a sigmoid probability, and training minimizes binary cross-entropy with gradient descent and backpropagation.

## Tech Stack

- Python
- Flask
- NumPy
- HTML Canvas
- Vanilla JavaScript
- CSS

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the app at:

```text
http://127.0.0.1:5000
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Project Structure

```text
app.py                 Flask app and API routes
mlp_visualizer/
  datasets.py          Synthetic dataset generators
  network.py           NumPy MLP and backpropagation logic
  trainer.py           Training state, snapshots, and decision grid
static/
  app.js               Frontend state, API calls, and canvas drawing
  styles.css           Application styling
templates/
  index.html           Main interface
```

## How It Works

1. The backend creates a dataset and initializes a small MLP.
2. The frontend requests the current training state from Flask.
3. Each training step runs forward propagation, computes binary cross-entropy, runs backpropagation, and updates parameters.
4. Flask returns the latest loss, accuracy, weights, biases, gradients, activations, and decision grid.
5. The browser redraws the model, boundary, history chart, and parameter values.

## License

This project is licensed under the MIT License.
