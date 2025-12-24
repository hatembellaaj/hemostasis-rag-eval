from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


@dataclass
class PaperLoadResult:
    docs: List[Document]
    source_file: str


def load_paper(path: Path) -> PaperLoadResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    # add filename for provenance
    for d in docs:
        d.metadata["source_file"] = path.name
    return PaperLoadResult(docs=docs, source_file=path.name)


@dataclass
class RegistryLoadResult:
    raw_tables: Dict[str, pd.DataFrame]
    source_file: str


def load_registry_excel(path: Path) -> RegistryLoadResult:
    xls = pd.ExcelFile(path)
    tables: Dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        tables[sheet] = df
    return RegistryLoadResult(raw_tables=tables, source_file=path.name)
