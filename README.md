# Risk-Calibrated Geometric Routing (RCGR) for Quantized Graph ANN Search

Risk-Calibrated Geometric Routing (RCGR) is a zero-parameter, statistically grounded stopping mechanism for graph-based Approximate Nearest Neighbor (ANN) search under quantization.

It replaces heuristic-based termination with a self-regulating framework, enabling:
- Zero user-tunable parameters (no delta, no target_recall, no ef_search)
- Query-adaptive termination via exponential volume-risk equivalence
- Automatic calibration from the dataset's quantization noise profile

## The Problem: Heuristic Stopping is Inefficient

Existing graph-based ANN search methods (e.g., HNSW, NSG, Vamana) perform search using greedy or beam-style traversal over a candidate queue. Termination is governed by heuristic conditions such as:
- fixed beam width / candidate pool size,
- local optimality (no closer neighbors),
- or distance-based pruning relative to current results.

These stopping rules are heuristic and query-agnostic, lacking a principled estimate of the probability that unexplored candidates may contain better neighbors. As a result, they often lead to inefficient or miscalibrated stopping behavior, especially under quantization or heterogeneous query difficulty.

### Limitations

1. **Ignoring Joint Risk**
   Existing methods reason about candidates individually or via local comparisons (e.g., best-first expansion), rather than modeling the **joint probability** that *any* remaining candidate in the queue could improve the result.

2. **Heuristic Termination Without Guarantees**
   Termination criteria (e.g., beam width or distance thresholds) are typically fixed or weakly adaptive, requiring manual tuning and offering no explicit control over recall or failure probability.

3. **Violation of Statistical Assumptions**
   Distribution-free methods such as Conformal Risk Control rely on exchangeability assumptions. However, graph traversal induces strong dependencies between visited candidates, violating these assumptions and leading to unreliable calibration.

---

## The Solution: Risk-Calibrated Geometric Routing (RCGR)

RCGR replaces heuristic stopping with a self-regulating exhaustion accumulator that maps statistical queue risk to topological search budget via exponential discounting.

The search maintains a floating-point exhaustion accumulator $c$, bounded by the graph's structural out-degree $M$. When the search expands an unpromising node, the step cost is dynamically discounted by the aggregate queue risk:

$$\Delta c = \exp\left(-R_{\mathcal{B}}\right)$$

where $R_{\mathcal{B}} = \sum_{x \in \mathcal{B}} S^+\!\left(\frac{\hat{d}^2(q,x)}{d_k^2(q)}\right)$ is the total expected number of hidden true neighbors in the queue.

### Key Idea
When the queue is statistically risky ($R_{\mathcal{B}} \gg 0$), the step cost shrinks exponentially, granting the search massive routing patience. When the queue is exhausted ($R_{\mathcal{B}} \to 0$), the step cost approaches $1.0$, and the search terminates in exactly $M$ unpromising steps. All parameters are derived from the dataset — the user supplies only `search(query, k)`.

---

## Mechanism

Each candidate in the queue has an approximate distance $\hat{d}^2(q,x)$ produced by RaBitQ quantization. The ratio $\hat{d}^2 / d^2$ between the approximate and exact distance is a random variable whose tail determines how often the estimator overestimates — the regime where a good neighbor might be mistakenly skipped. RCGR models this tail using the Generalized Pareto Distribution (GPD) via the Peaks-Over-Threshold (POT) framework.

**Tail fitting.** During offline calibration, estimation-error ratios are collected from probe queries. The sample mean of these ratios serves as the POT threshold $u$. The GPD is fitted to the exceedances above $u$ using Probability Weighted Moments (PWM, $\alpha$-moment convention). Since RaBitQ's quantization noise is bounded, the Pickands-Balkema-de Haan theorem guarantees $\xi < 0$, giving a finite upper endpoint $z_{\max} = u - \sigma/\xi$ beyond which the survival function is exactly zero.

**Risk evaluation.** At search time, each unvisited candidate's normalized distance $\hat{d}^2(q,x) / d_k^2(q)$ is passed through the fitted GPD survival function $S^+$, which returns the probability that the candidate's true distance is better than the current $k$-th neighbor. These probabilities are summed across the queue to produce $R_{\mathcal{B}}$. Because the queue is sorted by ascending approximate distance and $\xi < 0$ gives a finite endpoint, the summation breaks the moment a candidate exceeds $z_{\max}$, yielding $O(1)$ amortized evaluation.

**Routing.** The exhaustion accumulator $c$ governs termination via a piecewise update: if the expanded node improves the result set ($d(x) < d_k$), the search is navigating toward a better cluster and $c$ resets to $0$. Otherwise, the queue risk $R_{\mathcal{B}}$ is computed and the step cost $\exp(-R_{\mathcal{B}})$ is added to $c$. Search terminates when $c \geq M$ (the graph's structural out-degree).

## SIFT-1M Results

Evaluated on SIFT-1M (1M vectors, 128 dimensions) using the ann-benchmarks framework:

| Metric | Value |
|---|---|
| Build time | ~60 s (8-core Apple Silicon) |
| recall@10 | 0.78--0.80 |
| QPS (batch) | ~30,000 |
| User-tuned parameters | 0 |

GPD calibration: $\xi \approx -0.25$, $\sigma \approx 0.030$, $u \approx 1.0$, $z_{\max} \approx 1.12$.

Uses the standard [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) framework. The `evtq` algorithm wrapper is at `ann-benchmarks/ann_benchmarks/algorithms/evtq/`.
