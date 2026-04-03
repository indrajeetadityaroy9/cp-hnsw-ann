import numpy as np
import evtq
import time

rng = np.random.default_rng(42)
N = 5000; D = 128; K = 10; P = 200

centers = rng.standard_normal((50, D)).astype(np.float32) * 10
labels = rng.integers(0, 50, size=N)
vectors = (centers[labels] + rng.standard_normal((N, D)).astype(np.float32)).astype(np.float32)
queries = (centers[rng.integers(0, 50, size=P)] + rng.standard_normal((P, D)).astype(np.float32)).astype(np.float32)

gt_ids = np.empty((P, K), dtype=np.int64)
gt_dists = np.empty((P, K), dtype=np.float32)
for i in range(P):
    dists = np.sum((vectors - queries[i]) ** 2, axis=1)
    top_k = np.argpartition(dists, K)[:K]
    top_k = top_k[np.argsort(dists[top_k])]
    gt_ids[i] = top_k
    gt_dists[i] = dists[top_k]

idx = evtq.EVTQIndex(D)
idx.build(vectors)
idx.finalize()

# Detailed inspection of query 0
q = queries[0]
ids_ceil, dists_ceil = idx.search_graph_ceiling(q, 50)
ids_rcgr, dists_rcgr = idx.search(q, 50)

print("Query 0 — detailed inspection")
print(f"  Ground truth top-10 IDs:    {gt_ids[0].tolist()}")
print(f"  Ground truth top-10 dists²: {[f'{d:.1f}' for d in gt_dists[0]]}")
print()
print(f"  Graph ceiling returned {len(ids_ceil)} results")
print(f"    Top-10 IDs:    {ids_ceil[:10].tolist()}")
print(f"    Top-10 dists²: {[f'{d:.1f}' for d in dists_ceil[:10]]}")
print(f"    Worst dist²:   {dists_ceil[-1]:.1f}" if len(dists_ceil) > 0 else "")
print()
print(f"  RCGR returned {len(ids_rcgr)} results")
print(f"    Top-10 IDs:    {ids_rcgr[:10].tolist()}")
print(f"    Top-10 dists²: {[f'{d:.1f}' for d in dists_rcgr[:10]]}")
print()

# Check: are graph ceiling distances close to brute-force distances?
bf_dists_at_ceil_ids = np.array([np.sum((vectors[int(i)] - q) ** 2) for i in ids_ceil[:10]])
print(f"  BF distances at ceiling IDs: {[f'{d:.1f}' for d in bf_dists_at_ceil_ids]}")
print(f"  Match returned dists?        {np.allclose(dists_ceil[:10], bf_dists_at_ceil_ids, atol=0.1)}")
print()

# Overlap analysis
ceil_set = set(ids_ceil[:K].tolist())
rcgr_set = set(ids_rcgr[:K].tolist())
gt_set = set(gt_ids[0].tolist())
print(f"  GT ∩ ceiling@10: {len(gt_set & ceil_set)}/{K}")
print(f"  GT ∩ RCGR@10:    {len(gt_set & rcgr_set)}/{K}")
print(f"  Ceiling ∩ RCGR:  {len(ceil_set & rcgr_set)}/{K}")
print()

# Check if GT nodes appear ANYWHERE in top-50 ceiling results
ceil_50 = set(ids_ceil.tolist())
gt_in_top50 = len(gt_set & ceil_50)
print(f"  GT nodes found in ceiling top-50: {gt_in_top50}/{K}")

# Batch recall
hits = 0
for i in range(P):
    ids, _ = idx.search_graph_ceiling(queries[i], K)
    hits += len(set(ids.tolist()) & set(gt_ids[i].tolist()))
print(f"\n  Overall ceiling recall@{K}: {hits/(P*K):.4f}")

hits = 0
for i in range(P):
    ids, _ = idx.search(queries[i], K)
    hits += len(set(ids.tolist()) & set(gt_ids[i].tolist()))
print(f"  Overall RCGR recall@{K}:    {hits/(P*K):.4f}")
