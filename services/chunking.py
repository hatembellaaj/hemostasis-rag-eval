from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SplitMode = Literal["page", "heading"]
ChunkMode = Literal["fixed", "structure", "semantic"]


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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den_a = math.sqrt(sum(x * x for x in a))
    den_b = math.sqrt(sum(y * y for y in b))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _semantic_split(
    docs: List[Document],
    *,
    embedder: OpenAIEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
    similarity_threshold: float = 0.65,
) -> List[Document]:
    """Chunk documents by grouping semantically similar sentences.

    This keeps chunks near ``chunk_size`` characters but starts a new chunk when
    adjacent sentence embeddings diverge.
    """

    def _split_sentences(text: str) -> List[str]:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return parts if parts else [text]

    out: List[Document] = []
    for d in docs:
        sentences = _split_sentences(d.page_content)
        vectors = embedder.embed_documents(sentences)
        current: List[str] = []
        current_len = 0
        for sent, vec in zip(sentences, vectors):
            if not current:
                current.append(sent)
                current_len = len(sent)
                prev_vec = vec
                continue

            projected_len = current_len + 1 + len(sent)
            similarity = _cosine_similarity(prev_vec, vec)
            if projected_len > chunk_size or similarity < similarity_threshold:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    out.append(Document(page_content=chunk_text, metadata=dict(d.metadata)))

                if chunk_overlap > 0 and current:
                    # carry over the tail of the previous chunk to preserve context
                    overlap_sentences: List[str] = []
                    overlap_len = 0
                    for s in reversed(current):
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                        if overlap_len >= chunk_overlap:
                            break
                    current = overlap_sentences.copy()
                    current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)
                else:
                    current = []
                    current_len = 0

            current.append(sent)
            current_len += len(sent) + 1
            prev_vec = vec

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                out.append(Document(page_content=chunk_text, metadata=dict(d.metadata)))

    return out


def make_chunks(
    docs: List[Document],
    split_mode: SplitMode,
    chunk_mode: ChunkMode,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: Optional[str] = None,
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
    if chunk_mode == "semantic":
        embedder = OpenAIEmbeddings(model=embedding_model or "text-embedding-3-small")
        chunks = _semantic_split(
            processed,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
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
