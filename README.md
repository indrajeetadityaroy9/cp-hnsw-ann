# EVTQ: EVT-Calibrated Quantized Graph Search for Approximate Nearest Neighbor Search

Most graph-based approximate nearest neighbor (ANN) methods require tuning search parameters such as beam width, pruning thresholds, and termination conditions. EVTQ eliminates all search parameters by deriving them from the data at index build time. A Generalized Pareto Distribution (GPD) is fitted to distance estimation errors, producing calibrated thresholds with statistical guarantees. Combined with multi-bit RaBitQ quantization and AVX2 SIMD distance computation, EVTQ achieves competitive recall and throughput with zero manual tuning.

## Method

### Calibration

At build time, EVTQ collects errors between quantized distance estimates and true L2 distances over calibration queries drawn from the index's 2-hop graph neighborhoods. The error tail is modeled by a GPD fitted via Probability-Weighted Moments (Hosking & Wallis, 1987) using the median as the Peaks-Over-Threshold threshold. The fitted model produces a termination threshold γ at a self-normalizing return period (Pr[r > γ] = 1/m), with per-step distance slack computed analytically from segmentation-aware variance bounds.

### Quantization

Vectors are centered, rotated by a randomized Hadamard transform, and quantized:
- **1-bit**: sign quantization (RaBitQ)
- **2/4-bit**: coordinate-descent refinement maximizing cosine alignment between the code and rotated vector

### Search

Beam search evaluates neighbors in batches of 32 using AVX2 lookup-table kernels. The beam terminates when the best remaining candidate's estimated distance exceeds γ times the current k-th best exact distance. For multi-bit codes, a two-stage pipeline first computes a fast lower bound from the most significant bits, skipping full evaluation when the bound exceeds the current k-th distance.

### Graph Construction

The base-layer graph is built by NNDescent with adaptive convergence detection. Neighbor pruning uses α-CNG with parameters derived from the nearest-neighbor distance distribution. A BFS reorder from the graph hub improves cache locality.

## Usage

```python
import evtq

index = evtq.EVTQIndex(dim=128, bits=4)
index.build(vectors)
index.finalize()
ids, dists = index.search(query, k=10)
```

## Evaluation

```bash
python -m benchmarks --config configs/benchmark.yaml
```

Reports recall@10, QPS, build time, and memory for 1/2/4-bit configurations.

## References

1. Gao and Long. [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497). SIGMOD 2024.
2. Gao et al. [Extended RaBitQ: Practical and Asymptotically Optimal Quantization for High-Dimensional Vectors](https://arxiv.org/abs/2409.09913). SIGMOD 2025.
3. Malkov and Yashunin. [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320). TPAMI 2020.
