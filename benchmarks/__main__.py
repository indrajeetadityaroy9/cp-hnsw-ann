"""CLI entrypoint for ``python -m benchmarks``."""

import argparse
import json
from pathlib import Path

import yaml

from benchmarks.eval import run_benchmark


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmarks")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = cfg["data"]["datasets"]
    k = cfg["eval"]["k"]
    n_runs = cfg["eval"]["n_runs"]

    for repo_id in datasets:
        output = run_benchmark(repo_id, k, n_runs, output_dir)
        for algo in output["results"]:
            print(json.dumps({
                "dataset": output["metadata"]["dataset"],
                "algorithm": algo["algorithm"],
                "recall_at_10": algo["recall_at_10"],
                "qps": algo["qps"],
                "build_time_s": algo["build_time_s"],
                "memory_mb": algo["memory_mb"],
            }), flush=True)


if __name__ == "__main__":
    main()
