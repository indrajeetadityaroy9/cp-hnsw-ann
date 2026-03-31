import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil

import evtq
from benchmarks.datasets import load_dataset

BYTES_PER_MB = 1024 ** 2


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    eval_k = min(k, results.shape[1], ground_truth.shape[1])
    res = results[:, :eval_k]
    gt = ground_truth[:, :eval_k]
    hits = np.any(res[:, :, None] == gt[:, None, :], axis=2)
    return float(hits.sum(axis=1).mean()) / eval_k


def run_benchmark(repo_id: str,
                  k: int, n_runs: int, output_dir: Path) -> dict:
    ds = load_dataset(repo_id)
    base = ds["base"]
    queries = ds["queries"]
    gt = ds["groundtruth"].astype(np.int64)
    dim = ds["dim"]
    dataset_name = repo_id.split("/")[-1]

    gc.collect()
    rss_before = psutil.Process().memory_info().rss / BYTES_PER_MB
    t0 = time.perf_counter()

    index = evtq.EVTQIndex(dim=dim)
    index.build(base)
    index.finalize()

    build_time = time.perf_counter() - t0
    gc.collect()
    rss_after = psutil.Process().memory_info().rss / BYTES_PER_MB
    mem_mb = rss_after - rss_before

    def search_fn(batch):
        ids, _ = index.search_batch(batch, k=k)
        return np.asarray(ids)

    search_fn(queries)
    times = []
    t0 = time.perf_counter()
    ids = search_fn(queries)
    times.append(time.perf_counter() - t0)
    for _ in range(n_runs - 1):
        t0 = time.perf_counter()
        search_fn(queries)
        times.append(time.perf_counter() - t0)
    med_time = float(np.median(times))

    results = [{
        "algorithm": "evtq",
        "build_time_s": round(build_time, 2),
        "memory_mb": round(mem_mb, 1),
        "recall_at_10": round(recall_at_k(ids, gt, min(k, 10)), 4),
        "qps": round(len(queries) / med_time, 1),
    }]

    del index
    gc.collect()

    output = {
        "metadata": {
            "dataset": dataset_name,
            "n_base": len(base),
            "n_queries": len(queries),
            "dim": dim,
            "k": k,
            "n_runs": n_runs,
        },
        "results": results,
    }

    outfile = output_dir / f"{dataset_name}_results.json"
    with outfile.open("w") as f:
        json.dump(output, f, indent=2)

    return output
