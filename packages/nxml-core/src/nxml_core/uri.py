from __future__ import annotations

from pathlib import Path


def resolve_model_uri(uri: str) -> Path:
    """Resolve a model URI to a local filesystem path.

    Supported schemes:
      - ``hf:owner/repo/path/to/file.pt`` — fetched via huggingface_hub.
      - ``file:./path`` or ``file:/abs/path`` — local file (scheme stripped).
      - bare path (e.g. ``/abs/path`` or ``./relative.pt``) — returned as Path.
    """
    if uri.startswith("hf:"):
        spec = uri[len("hf:") :]
        parts = spec.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                f"hf URI must be 'hf:owner/repo/path/to/file', got {uri!r}"
            )
        owner, repo, filename = parts[0], parts[1], parts[2]
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=f"{owner}/{repo}", filename=filename))

    if uri.startswith("file:"):
        return Path(uri[len("file:") :])

    return Path(uri)
