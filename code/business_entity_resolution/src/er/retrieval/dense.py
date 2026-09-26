"""Dense embedding retrieval with OpenVINO Intel Arc GPU acceleration (F05 Phase D / E04).

Implements:
- OpenVINO GPU inference on Intel Arc 140T GPU (16GB).
- Attention-mask aware mean pooling + L2 normalization (fixes unmasked demo defect).
- Faiss IndexFlatIP exact cosine search.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import openvino as ov
import torch
from optimum.intel.openvino import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import faiss


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Mean pooling taking attention mask into account for correct averaging."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled = (sum_embeddings / sum_mask).detach().cpu().numpy()
    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (pooled / norms).astype(np.float32)


class ArcGPUDenseRetriever:
    def __init__(self, model_dir: str | Path, device: str = "GPU"):
        self.model_dir = Path(model_dir)
        core = ov.Core()
        available = core.available_devices
        self.device = device if device in available else "CPU"
        ov_config = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "AUTO",
            "INFERENCE_PRECISION_HINT": "f16",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = OVModelForFeatureExtraction.from_pretrained(
            str(self.model_dir), device=self.device, ov_config=ov_config
        )

    def encode(self, texts: list[str], batch_size: int = 512, max_length: int = 128) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**encoded)
                # outputs.last_hidden_state has shape (batch_size, seq_len, hidden_dim)
                hidden_state = outputs.last_hidden_state
                pooled = mean_pooling(hidden_state, encoded["attention_mask"])
                all_embeddings.append(pooled)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 384), dtype=np.float32)


def search_dense(query_embeddings: np.ndarray,
                 pool_embeddings: np.ndarray,
                 pool_ids: list[str],
                 query_ids: list[str],
                 top_k: int = 50) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Exact Faiss IndexFlatIP search using normalized embeddings."""
    dim = pool_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(pool_embeddings)

    sims, indices = index.search(query_embeddings, top_k)
    pool_ids_arr = np.asarray(pool_ids, dtype=str)

    cands_out: dict[str, list[str]] = {}
    scores_out: dict[str, list[float]] = {}

    for q_idx, qid in enumerate(query_ids):
        p_sub = pool_ids_arr[indices[q_idx]]
        cands_out[qid] = [str(x) for x in p_sub]
        scores_out[qid] = [float(x) for x in sims[q_idx]]

    return cands_out, scores_out
