# EVTQ: Calibrated Termination for Quantized Graph Search via Extreme Value Theory

Graph-based approximate nearest neighbor search attains strong recall-throughput tradeoffs by combining sparse graph traversal with cheap quantized distance estimates, but deployed systems still rely on manually tuned search cutoffs that must be swept separately for each dataset and operating point. EVTQ removes this tuning loop by calibrating the termination rule directly from the built index. The method combines 4-bit RaBitQ-style vector quantization, NNDescent-style graph refinement, and an extreme-value model of quantization overestimation: after index construction, EVTQ samples estimation ratios from graph neighborhoods, fits the right tail with a Generalized Pareto Distribution using probability-weighted moments, and extracts a stopping multiplier that is then used at query time to terminate graph exploration without an external search parameter.

## Usage

```python
import numpy as np
import evtq

vectors = np.random.randn(10000, 128).astype(np.float32)
query = vectors[0]

index = evtq.EVTQIndex(dim=128)
index.build(vectors)
index.finalize()

ids, distances = index.search(query, k=10)
```
