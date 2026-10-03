# DynaFlux Algorithm Notes

## 1. Multi-period inputs

For a traffic tensor `X ∈ R^(T×N×C)`, three windows of length `L=12` are built for every forecast origin `t`:

- recent hour: `X[t-L:t]`
- aligned previous day: `X[t-288-L:t-288]`
- aligned previous week: `X[t-2016-L:t-2016]`

The target is `Y = X[t:t+12]`.

## 2. Temporal encoding

Each period branch fuses projected flow values, time-of-day, day-of-week, node embeddings and step embeddings. Multi-head temporal self-attention is then applied independently to every node sequence. The default setup uses hidden size 64, three encoder layers and four attention heads.

## 3. Global dynamic flow graph

Encoded node sequences are flattened. For nodes `i,j`:

`d(i,j) = ||x_i - x_j||_2`

`s(i,j) = 1 / (1 + d(i,j))`

A symmetric normalized graph Laplacian is formed:

`L = I - D^(-1/2) S D^(-1/2)`

The eigenvectors associated with the smallest `K` eigenvalues form a spectral embedding. K-means produces node groups, and co-membership becomes the global association matrix `A_G`.

## 4. Local spatial topology graph

Road distances are converted to edge affinities with a Gaussian kernel and weak edges are filtered by a threshold. Louvain communities provide regional structure. Topology-masked graph attention learns the local association matrix `A_L`.

## 5. Multi-scale fusion

Global and local matrices interact element-wise:

`A_M = softmax(A_G ⊙ A_L)`

`A_M` then aggregates node features into the multi-scale spatial representation `X_MS`.

## 6. Adaptive gated convolution

Residual traffic features and `X_MS` are fused by an MLP into `X_ST`. Two temporal branches are applied:

`X1 = Conv_d(X_ST)`

`X2 = Conv(X_ST) ⊙ (1 + ε), ε ~ U(-0.1, 0.1)`

`G = sigmoid(X1 + X2)`

`X_AC = G ⊙ X1 + (1-G) ⊙ X2`

The implementation additionally uses a residual connection and LayerNorm for stable optimization.

## 7. Multi-horizon regression

The final node features from the hour/day/week branches are concatenated and passed to an MLP regression head that predicts the next 12 frames. Training uses Huber loss, Adam with initial learning rate `1e-3`, batch size `16`, and gradient clipping at `5.0`.
