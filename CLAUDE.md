# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

EVTQ — a C++23 ANN search library implementing Fortified Queue-Risk-Calibrated Termination (QRCT). Combines 4-bit RaBitQ vector quantization, NNDescent graph construction, and GPD-based calibrated termination for recall-guaranteed search without manual hyperparameter tuning.

## Build & Run

```bash
# Install (builds C++ extension via CMake + pybind11)
pip install -e .
```

Requires: Python 3.12+, C++23 (GCC 13+ / Clang 17+), AVX2+FMA, OpenMP.

## Architecture

Three-stage pipeline: **Build** → **Finalize** → **Search**.

| Stage | What happens | Key files |
|-------|-------------|-----------|
| Build | FHT rotation, 4-bit CAQ encoding, NNDescent graph + alpha-pruning, BFS reorder | `encoder.hpp`, `graph_build.hpp`, `graph.hpp` |
| Finalize | QRCT calibration: POT threshold, constrained PWM GPD fit (ξ ∈ [-1,0)), binary-search δ for target recall | `calibration.hpp`, `estimator.hpp` |
| Search | Beam search with GPD queue risk summation, monotonic truncation at survival=0 | `search.hpp`, `fastscan.hpp` |

`index.hpp` — top-level `Index<D>` template aggregating all components.
`bindings.cpp` — pybind11 Python interface.
`core.hpp` — shared types (`NodeId`, `SearchResult`), SIMD alignment, `gpd_survival()`.

## Theory Source

`new_mechanism.md` is the **single source of truth** for the algorithm specification. Every formula, case, and step defined there must have a direct code counterpart. Both ξ = 0 and ξ ≠ 0 branches must exist if the theory defines them.

## Code Philosophy

- **Research codebase with controlled inputs.** No input validation, no edge-case guards, no defensive logic, no backward compatibility, no feature flags.
- **Single canonical execution path.** No optional modes, no parallel implementations, no dead code.
- **Zero magic numbers.** Every literal must trace to a theorem, hardware constant, or structural property. Aliasing arbitrary values via `constexpr` is still a magic number.
- **No unit tests.** Benchmarks are the validation tool.
- **KISS.** If a value is structurally guaranteed upstream, do not re-check it downstream. Consolidate duplicated logic. Delete dead code rather than commenting it out.
