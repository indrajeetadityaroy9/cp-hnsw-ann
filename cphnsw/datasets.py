"""Dataset loaders for ANN benchmarks via Hugging Face Hub."""

import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files


def _download(repo_id, filename):
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))


def _load_fvecs(path):
    raw = np.fromfile(path, dtype=np.float32)
    dim = raw[:1].view(np.int32)[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy()


def _load_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[0])
    return raw.reshape(-1, k + 1)[:, 1:].copy()


def _extract(archive_path, suffix):
    extract_dir = archive_path.parent / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith(suffix):
                target = extract_dir / Path(m.name).name
                target.write_bytes(tar.extractfile(m).read())
                return target


def _find(files, suffix):
    return next(f for f in files if f.endswith(suffix))


def load_dataset(repo_id):
    files = list_repo_files(repo_id, repo_type="dataset")

    if any(f.endswith(".fvecs") for f in files):
        base = _load_fvecs(_download(repo_id, _find(files, "_base.fvecs")))
        queries = _load_fvecs(_download(repo_id, _find(files, "_query.fvecs")))
        gt = _load_ivecs(_download(repo_id, _find(files, "_groundtruth.ivecs")))
    elif any(f.endswith(".tar.gz") for f in files):
        archive = _download(repo_id, _find(files, ".tar.gz"))
        base = _load_fvecs(_extract(archive, "_base.fvecs"))
        queries = _load_fvecs(_extract(archive, "_query.fvecs"))
        gt = _load_ivecs(_extract(archive, "_groundtruth.ivecs"))
    else:
        base = np.load(_download(repo_id, _find(files, "base.npy")), allow_pickle=False).astype(np.float32)
        queries = np.load(_download(repo_id, _find(files, "queries.npy")), allow_pickle=False).astype(np.float32)
        gt = np.load(_download(repo_id, _find(files, "groundtruth.npy")), allow_pickle=False).astype(np.int32)

    return {"base": base, "queries": queries, "groundtruth": gt, "dim": base.shape[1]}
