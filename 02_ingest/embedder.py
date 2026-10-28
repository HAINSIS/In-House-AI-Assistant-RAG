"""Local text embeddings: zero API tokens, nothing leaves the machine.

bge-m3 (BAAI, 568M params, 1024-dim) because the corpus and the questions
mix English and Chinese, and it reads up to 8192 tokens. Runs in fp32: the
Pascal-generation GPUs (e.g. GTX 10xx) have no fast fp16.
Vectors are L2-normalised, so cosine similarity is a dot product.
"""
from __future__ import annotations

import os
from pathlib import Path


def _cached(model_name: str) -> bool:
    """True if the model is in the Hugging Face cache (default layout).
    If the cache is incomplete, delete that folder and run again to re-download."""
    hub = os.environ.get("HF_HUB_CACHE") or \
        str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snapshots = Path(hub) / ("models--" + model_name.replace("/", "--")) / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16,
                 hf_endpoint: str | None = None):
        if hf_endpoint:
            # huggingface.co is slow or blocked here; the mirror is reached
            # directly, not through the data-limited VPN proxy.
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
            host = hf_endpoint.split("//")[-1].split("/")[0]
            os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), host]))
        if _cached(model_name):
            # Once downloaded, never touch the network: without this every start
            # makes ~10 hub requests (update and weight-conversion checks).
            # Must be set before huggingface_hub is imported.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)
        return vecs.tolist()
