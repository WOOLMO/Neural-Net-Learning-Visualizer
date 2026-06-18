const els = {
  step: document.querySelector("#step"),
  loss: document.querySelector("#loss"),
  accuracy: document.querySelector("#accuracy"),
  boundary: document.querySelector("#boundary"),
  network: document.querySelector("#network"),
  history: document.querySelector("#history"),
  parameters: document.querySelector("#parameters"),
  datasetDescription: document.querySelector("#datasetDescription"),
  dataset: document.querySelector("#dataset"),
  architecture: document.querySelector("#architecture"),
  learningRate: document.querySelector("#learningRate"),
  learningRateLabel: document.querySelector("#learningRateLabel"),
  seed: document.querySelector("#seed"),
  stepOnce: document.querySelector("#stepOnce"),
  runToggle: document.querySelector("#runToggle"),
  reset: document.querySelector("#reset"),
};

let state = null;
let running = false;
let timer = null;

const datasetDescriptions = {
  moons: "The selected dataset is two interleaving half-moon shapes. The MLP is trained to separate points that cannot be split by one straight line, so the hidden layers must bend the decision boundary around the arcs.",
  xor: "The selected dataset is XOR. Points in opposite diagonal corners share the same class. A linear model fails here, so the MLP must combine hidden neuron features to carve the plane into alternating regions.",
  spiral: "The selected dataset is a two-arm spiral. This is the hardest option because the classes wind around each other. Training shows how weights and biases gradually twist the decision boundary into a nonlinear shape.",
};

async function api(path, body = null) {
  const options = body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : {};
  const response = await fetch(path, options);
  return response.json();
}

function resetPayload() {
  return {
    dataset: els.dataset.value,
    hiddenLayers: els.architecture.value.split(",").map(Number),
    learningRate: Number(els.learningRate.value),
    seed: Number(els.seed.value),
  };
}

async function refresh(nextState = null) {
  state = nextState || await api("/api/state");
  els.step.textContent = state.step;
  els.loss.textContent = state.loss.toFixed(4);
  els.accuracy.textContent = `${Math.round(state.accuracy * 100)}%`;
  els.datasetDescription.textContent = datasetDescriptions[state.datasetName || els.dataset.value] || datasetDescriptions.moons;
  drawBoundary();
  drawNetwork();
  drawHistory();
  drawParameters();
}

async function train(steps = 1) {
  await refresh(await api("/api/train", { steps }));
}

function clear(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  return ctx;
}

function project(point, canvas) {
  const margin = 34;
  const scaleX = (canvas.width - margin * 2) / 5.2;
  const scaleY = (canvas.height - margin * 2) / 5.2;
  return [
    margin + (point[0] + 2.6) * scaleX,
    canvas.height - margin - (point[1] + 2.6) * scaleY,
  ];
}

function classColor(v, alpha = 1) {
  const r = Math.round(240 * v + 68 * (1 - v));
  const g = Math.round(111 * v + 166 * (1 - v));
  const b = Math.round(103 * v + 229 * (1 - v));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function drawBoundary() {
  const canvas = els.boundary;
  const ctx = clear(canvas);
  const { xs, ys, z } = state.grid;
  const cellW = canvas.width / xs.length;
  const cellH = canvas.height / ys.length;

  for (let row = 0; row < ys.length; row++) {
    for (let col = 0; col < xs.length; col++) {
      ctx.fillStyle = classColor(z[row][col], 0.58);
      ctx.fillRect(col * cellW, canvas.height - (row + 1) * cellH, cellW + 1, cellH + 1);
    }
  }

  state.dataset.x.forEach((point, index) => {
    const [x, y] = project(point, canvas);
    const label = state.dataset.y[index];
    ctx.beginPath();
    ctx.arc(x, y, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = label ? "#f06f67" : "#62a8e5";
    ctx.fill();
    ctx.strokeStyle = "#101214";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function drawNetwork() {
  const canvas = els.network;
  const ctx = clear(canvas);
  const sizes = state.network.layerSizes;
  const acts = state.network.sampleActivations;
  const layerGap = canvas.width / (sizes.length + 1);

  const positions = sizes.map((size, layer) => {
    const x = layerGap * (layer + 1);
    const gap = Math.min(68, (canvas.height - 90) / Math.max(1, size - 1));
    const start = canvas.height / 2 - gap * (size - 1) / 2;
    return Array.from({ length: size }, (_, node) => [x, start + node * gap]);
  });

  state.network.weights.forEach((matrix, layer) => {
    matrix.forEach((row, from) => {
      row.forEach((weight, to) => {
        const [x1, y1] = positions[layer][from];
        const [x2, y2] = positions[layer + 1][to];
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = weight >= 0 ? `rgba(89,201,141,${Math.min(0.9, Math.abs(weight) / 2 + 0.15)})` : `rgba(240,111,103,${Math.min(0.9, Math.abs(weight) / 2 + 0.15)})`;
        ctx.lineWidth = Math.max(0.7, Math.min(5, Math.abs(weight) * 2.3));
        ctx.stroke();
      });
    });
  });

  positions.forEach((layerPositions, layer) => {
    layerPositions.forEach(([x, y], node) => {
      const activation = acts[layer]?.[node] ?? 0;
      ctx.beginPath();
      ctx.arc(x, y, 17, 0, Math.PI * 2);
      ctx.fillStyle = classColor((activation + 1) / 2, 0.95);
      if (layer === sizes.length - 1) ctx.fillStyle = classColor(activation, 0.95);
      ctx.fill();
      ctx.strokeStyle = "#f4f1e8";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#101214";
      ctx.font = "10px Consolas";
      ctx.textAlign = "center";
      ctx.fillText(Number(activation).toFixed(2), x, y + 3);
    });
  });

  ctx.fillStyle = "#9fa8a5";
  ctx.font = "13px Segoe UI";
  ctx.textAlign = "center";
  sizes.forEach((size, i) => ctx.fillText(i === 0 ? "input" : i === sizes.length - 1 ? "output" : `hidden ${i}`, layerGap * (i + 1), 26));
}

function drawHistory() {
  const canvas = els.history;
  const ctx = clear(canvas);
  const history = state.history;
  ctx.strokeStyle = "#30383d";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = 28 + i * 48;
    ctx.beginPath();
    ctx.moveTo(36, y);
    ctx.lineTo(canvas.width - 16, y);
    ctx.stroke();
  }
  if (history.length < 2) return;

  plotLine(ctx, history.map(h => h.loss), "#e5c45d", Math.max(...history.map(h => h.loss), 1));
  plotLine(ctx, history.map(h => h.accuracy), "#59c98d", 1);
  ctx.fillStyle = "#e5c45d";
  ctx.fillText("loss", 42, 20);
  ctx.fillStyle = "#59c98d";
  ctx.fillText("accuracy", 92, 20);
}

function plotLine(ctx, values, color, maxValue) {
  const left = 36;
  const top = 28;
  const width = ctx.canvas.width - 52;
  const height = ctx.canvas.height - 48;
  ctx.beginPath();
  values.forEach((value, i) => {
    const x = left + (i / Math.max(1, values.length - 1)) * width;
    const y = top + height - (value / maxValue) * height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.stroke();
}

function drawParameters() {
  const html = state.network.weights.map((matrix, layer) => {
    const weights = matrix.flat().map(v => `<span class="chip">w ${signed(v)}</span>`).join("");
    const biases = state.network.biases[layer].map(v => `<span class="chip">b ${signed(v)}</span>`).join("");
    const grads = state.network.gradWeights[layer].flat().map(v => `<span class="chip">dw ${signed(v)}</span>`).join("");
    return `<div class="param-row"><strong>Layer ${layer + 1}</strong><div class="chips">${weights}${biases}${grads}</div></div>`;
  }).join("");
  els.parameters.innerHTML = html;
}

function signed(value) {
  return `${value >= 0 ? "+" : ""}${Number(value).toFixed(3)}`;
}

els.learningRate.addEventListener("input", () => {
  els.learningRateLabel.textContent = Number(els.learningRate.value).toFixed(2);
});

els.stepOnce.addEventListener("click", () => train(1));
els.reset.addEventListener("click", async () => refresh(await api("/api/reset", resetPayload())));
els.runToggle.addEventListener("click", () => {
  running = !running;
  els.runToggle.textContent = running ? "Pause" : "Run";
  if (running) timer = setInterval(() => train(4), 180);
  else clearInterval(timer);
});

refresh();
