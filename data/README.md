# Data layout

Place raw benchmark files here:

```text
data/raw/
├── PEMS08.npz
├── PEMS08.csv
├── PEMS04.npz
└── PEMS04.csv
```

The NPZ tensor is expected to use `[time, node, feature]`. The default configuration reads channel `0` as traffic flow. The edge CSV accepts `from,to,cost` or equivalent source/target/distance column names.

Raw datasets are intentionally not bundled in this repository.
