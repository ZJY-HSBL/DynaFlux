# DynaFlux — English Guide

DynaFlux is a multi-scale dynamic graph framework for multi-step traffic-flow forecasting. It is designed to jointly represent short-term dynamics, daily periodicity and weekly periodicity while combining long-range flow similarity with local road topology.

## Architecture

The input contains three aligned windows: the latest hour, the corresponding hour one day earlier, and the corresponding hour one week earlier. Each window has 12 five-minute frames and is processed by an isomorphic temporal branch. The temporal encoder combines projected flow values, time-of-day features, day-of-week features, node-adaptive embeddings and multi-head self-attention.

The global graph branch flattens each node's temporal feature sequence, computes pairwise Euclidean distance, and converts distance to similarity with `1/(1+d)`. A normalized graph Laplacian is eigendecomposed, and the eigenvectors associated with the smallest eigenvalues are clustered by K-means to form a long-range flow-association graph.

The local graph branch builds a weighted road-topology graph from edge data. Louvain community discovery supplies regional structure, while topology-masked graph attention learns adaptive neighbor importance. Global and local association matrices interact element-wise and are normalized by Softmax to obtain a multi-scale dynamic graph for node aggregation.

The adaptive convolution module first fuses residual traffic features and graph-aggregated features with an MLP. It then applies a dilated causal convolution and a standard causal convolution in parallel. The gate is `G = sigmoid(X1 + X2)`, and the gated output is `G ⊙ X1 + (1-G) ⊙ X2`, followed by a residual connection.

The three period branches are concatenated at the node level and mapped by a regression head to the next 12 five-minute frames.

## Training

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train.py --config configs/pems08.yaml
```

Artifacts are written to `outputs/pems08/`, including best/last checkpoints, training history, evaluation metrics, scaling statistics and the cached spatial graph.

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/pems08.yaml \
  --checkpoint outputs/pems08/best.pt \
  --graph outputs/pems08/graph.npz \
  --scaler outputs/pems08/scaler.npz
```

Metrics include MAE, RMSE, MAPE and R² at 15, 30, 45 and 60 minutes, plus averages over the full 12-step forecast horizon.

## Smoke test

```bash
python scripts/demo_synthetic.py
pytest -q
```

The synthetic demo generates traffic series with daily and weekly periodic components and validates the full forward path without external datasets.
