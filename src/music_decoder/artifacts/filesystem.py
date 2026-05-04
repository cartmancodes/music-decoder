from __future__ import annotations

from pathlib import Path


class FilesystemArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        target = (self.root / key).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as e:
            raise ValueError(f"path traversal not allowed: {key!r}") from e
        return target

    def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def read(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).exists()
        except ValueError:
            return False

    def path_for(self, key: str) -> Path:
        return self._resolve(key)
