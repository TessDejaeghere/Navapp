"""Build (or rebuild) the chatbot index: research corpus + letters uploaded from Navapp.

Writes a new version under data/index/vNNN/ and switches data/index/CURRENT to it.
Embeddings are cached, so only new or changed fragments are sent to OpenRouter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.indexer import rebuild_index  # noqa: E402

if __name__ == "__main__":
    print(rebuild_index(progress=lambda f, m: print(f"  {f:4.0%} {m}", flush=True)))
