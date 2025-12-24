from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SplitMode = Literal["page", "heading"]
ChunkMode = Literal["fixed", "structure"]


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_HEADING_RE = re.compile(r"^(?:[A-Z][A-Z \-]{3,}|(?:\d+(\.\d+)*\s+)[A-Z].{0,80})$")


def split_by_heading(docs: List[Document]) -> List[Document]:
    """
    Heuristic split: within each doc, split blocks when a line looks like a heading.
    This is deliberately simple for robustness across PDFs.
    """
    out: List[Document] = []
    for d in docs:
        text = d.page_content or ""
        lines = [ln.strip() for ln in text.splitlines()]
        blocks: List[str] = []
        cur: List[str] = []
        for ln in lines:
            if _HEADING_RE.match(ln) and cur:
                blocks.append("\n".join(cur).strip())
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            blocks.append("\n".join(cur).strip())

        for j, blk in enumerate(blocks):
            nd = Document(page_content=blk, metadata=dict(d.metadata))
            nd.metadata["split_unit"] = "heading"
            nd.metadata["split_idx"] = j
            out.append(nd)
    return out


def make_chunks(
    docs: List[Document],
    split_mode: SplitMode,
    chunk_mode: ChunkMode,
    chunk_size: int,
    chunk_overlap: int,
    do_clean: bool = True,
) -> List[Document]:
    # clean + add minimal fields
    processed: List[Document] = []
    for d in docs:
        text = d.page_content or ""
        if do_clean:
            text = clean_text(text)
        nd = Document(page_content=text, metadata=dict(d.metadata))
        processed.append(nd)

    if split_mode == "heading":
        processed = split_by_heading(processed)
    else:
        for d in processed:
            d.metadata["split_unit"] = "page"

    # chunking
    if chunk_mode == "structure":
        # still use RecursiveCharacterTextSplitter but rely on separators to preserve paragraphs better
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks = splitter.split_documents(processed)
    # assign chunk ids
    for i, c in enumerate(chunks):
        c.metadata["chunk_idx"] = i
    return chunks
