from pathlib import Path

import pytest

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.artifacts.filesystem import FilesystemArtifactStore


def test_store_writes_and_reads(tmp_path: Path):
    store: ArtifactStore = FilesystemArtifactStore(root=tmp_path)
    key = store.put("uploads/1/source.wav", b"hello world")
    assert key == "uploads/1/source.wav"
    assert store.exists(key)
    assert store.read(key) == b"hello world"


def test_store_path_for_key(tmp_path: Path):
    store = FilesystemArtifactStore(root=tmp_path)
    key = store.put("uploads/1/source.wav", b"x")
    assert store.path_for(key) == tmp_path / "uploads/1/source.wav"


def test_store_open_for_write(tmp_path: Path):
    store = FilesystemArtifactStore(root=tmp_path)
    target = store.path_for("derived/2/midi.mid")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"midi bytes")
    assert store.exists("derived/2/midi.mid")


def test_store_rejects_traversal(tmp_path: Path):
    store = FilesystemArtifactStore(root=tmp_path)
    with pytest.raises(ValueError, match="path traversal"):
        store.put("../escape.txt", b"x")
