"""
Two-Tower embedding store — separate from embedding_model/embedding_store.py.

Holds the 64-dim vectors produced by the tt_model. Updated after every
training step so tt_inference can score items without re-running the model.

Both dicts are plain Python dicts (not thread-local) because they are read
by the retrieval coroutine and written by the training coroutine, but only
ONE thread runs the asyncio event loop so there is no concurrent mutation.
"""

from typing import Dict
import torch

# user_id (str) → 64-dim L2-normalised tensor
tt_user_embeddings: Dict[str, torch.Tensor] = {}

# item_id (str) → 64-dim L2-normalised tensor
tt_item_embeddings: Dict[str, torch.Tensor] = {}
