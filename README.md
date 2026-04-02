# Queue-Risk-Calibrated Termination (QRCT) for Quantized Graph ANN Search

Queue-Risk-Calibrated Termination (QRCT) is a statistically grounded stopping criterion for graph-based Approximate Nearest Neighbor (ANN) search under quantization.

It replaces heuristic-based termination with a risk-calibrated framework, enabling:
- Improved efficiency
- Controlled recall guarantees
- Automatic parameter calibration


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

## The Solution: Queue-Risk-Calibrated Termination (QRCT)

QRCT replaces heuristic stopping with a total risk threshold $\delta$.

The search terminates only when the aggregate risk of the candidate queue $\mathcal{B}$ is sufficiently small:

$$
R_{\mathcal{B}}(q) = \sum_{x \in \mathcal{B}(q)} S^+ \left( \frac{\hat{d}^2(q,x)}{d_k^2(q)} \right) \leq \delta
$$

### Key Idea
Instead of relying on heuristic queue exhaustion or distance comparisons, QRCT explicitly estimates the probability that any remaining candidate could outperform the current $k$-th neighbor. Search stops only when this **total residual risk** falls below a user-specified threshold.

---

## Mechanism

Each candidate in the queue has an approximate distance $\hat{d}^2(q,x)$ produced by RaBitQ quantization. The ratio $\hat{d}^2 / d^2$ between the approximate and exact distance is a random variable whose tail determines how often the estimator overestimates — the regime where a good neighbor might be mistakenly skipped. QRCT models this tail using the Generalized Pareto Distribution (GPD) via the Peaks-Over-Threshold (POT) framework.

**Tail fitting.** During offline calibration, estimation-error ratios are collected from probe queries. The sample mean of these ratios serves as the POT threshold $u$ — a parameter-free choice that separates the bulk distribution from the overestimation tail. The GPD is fitted to the exceedances above $u$ using Probability Weighted Moments (PWM) with a single constraint: the shape parameter $\xi \geq -1$, which prevents the GPD density from diverging at its finite endpoint. Since RaBitQ's quantization noise is $O(1/\sqrt{D})$-bounded, the Pickands-Balkema-de Haan theorem guarantees $\xi < 0$, meaning the GPD has a finite upper endpoint beyond which the survival function is exactly zero.

**Risk evaluation.** At search time, each unvisited candidate's normalized distance $\hat{d}^2(q,x) / d_k^2(q)$ is passed through the fitted GPD survival function $S^+$, which returns the probability that the candidate's true distance is better than the current $k$-th neighbor. These probabilities are summed across the queue to produce $R_B(q)$. Because the queue is sorted by ascending approximate distance and $\xi < 0$ gives a finite endpoint, the survival values decrease monotonically to zero — the sum can be truncated the moment a candidate exceeds the GPD's upper bound, yielding $O(M_\epsilon)$ output-sensitive evaluation.

**Calibration.** Rather than modeling spatial dependence among graph neighbors, all modeling error — union bound looseness, threshold choice, PWM estimation noise — is absorbed by a single empirical step: binary search over $\delta$ on probe queries to find the maximum risk budget that achieves a target recall $R_\text{target}$. The search converges when the interval width falls below $1/k$, since recall is discrete (integer hits divided by $k$).
