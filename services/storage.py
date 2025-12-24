from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() else default


@dataclass(frozen=True)
class Settings:
    root: Path
    artifacts_dir: Path
    chroma_dir: Path
    data_papers_dir: Path
    data_registry_dir: Path

    embedding_model: str
    chat_model: str

    @staticmethod
    def load() -> "Settings":
        root = Path(__file__).resolve().parents[1]
        artifacts_dir = Path(_env("ARTIFACTS_DIR", str(root / "artifacts")))
        chroma_dir = Path(_env("CHROMA_DIR", str(root / "chroma_db")))
        data_papers_dir = root / "data" / "papers"
        data_registry_dir = root / "data" / "registry"

        return Settings(
            root=root,
            artifacts_dir=artifacts_dir,
            chroma_dir=chroma_dir,
            data_papers_dir=data_papers_dir,
            data_registry_dir=data_registry_dir,
            embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=_env("CHAT_MODEL", "gpt-4.1-mini"),
        )


def ensure_dirs(settings: Settings) -> None:
    (settings.artifacts_dir / "pivot").mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "stats").mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "runs").mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.data_papers_dir.mkdir(parents=True, exist_ok=True)
    settings.data_registry_dir.mkdir(parents=True, exist_ok=True)


def new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
