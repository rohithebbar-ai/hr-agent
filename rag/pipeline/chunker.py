# rag/pipeline/chunker.py

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict


MAX_NARRATIVE_TOKENS = 400
OVERLAP_SENTENCES = 2
SHORT_THRESHOLD = 300


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    element_type: str
    section_title: str
    section_path: str       # "Benefits > Health Insurance > Eligibility"
    page_hint: int
    is_table: bool
    token_count: int

    # Populated by pipeline orchestrator before embedding
    embedding: list = field(default_factory=list)
    sparse_embedding: object = None
    filename: str = ""
    file_type: str = ""
    document_title: str = ""
    upload_date: str = ""
    tenant_id: str = "vanaciprime"
    document_type: str = "handbook"


def _build_path(section_stack: Dict[int, str]) -> str:
    """
    Build hierarchical breadcrumb from section stack.

    section_stack = {1: "Employee Benefits", 2: "Health Insurance", 3: "Eligibility"}
    → "Employee Benefits > Health Insurance > Eligibility"

    This is what makes retrieval context-aware. A chunk about "30 days"
    under "Benefits > Health Insurance > Eligibility" is very different
    from a chunk about "30 days" under "Leave > Probationary Period".
    """
    if not section_stack:
        return "Document"
    sorted_levels = sorted(section_stack.keys())
    parts = [section_stack[level] for level in sorted_levels]
    return " > ".join(parts)


def _is_markdown_table(text: str) -> bool:
    """
    Safety net: detect markdown tables even if element_type != Table.
    LlamaParse usually classifies tables correctly but occasionally
    returns them as NarrativeText. Check actual content structure.
    Unstructured fast never produces markdown tables, so this only
    triggers on LlamaParse output edge cases.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return (
        len(lines) >= 2
        and lines[0].startswith("|")
        and lines[0].endswith("|")
        and all("|" in line for line in lines[:3])
    )


def chunk_document(loaded_doc) -> List[Chunk]:
    """
    Element-aware chunking with hierarchical section tracking.

    Rules:
    - Title          → update section_stack by heading_level, no chunk
    - Header/Footer  → discard
    - Table          → keep whole always (including markdown tables from LlamaParse)
    - ListItem       → buffer, group with preceding NarrativeText
    - NarrativeText  → short: keep whole, long: split with 2-sentence overlap
    """
    chunks = []
    chunk_index = 0
    doc_title = loaded_doc.title or loaded_doc.filename

    # section_stack tracks the current heading hierarchy
    # key = heading level (1, 2, 3), value = heading text
    # Example: {1: "Employee Benefits", 2: "Health Insurance"}
    section_stack: Dict[int, str] = {}

    current_section = "Document"   # Latest title text (for section_title field)
    pending_context = None         # NarrativeText before a list
    list_buffer = []               # Accumulated ListItems

    for element in loaded_doc.elements:
        el_type = element.element_type

        # ── Title: update section hierarchy, no chunk created ──
        if el_type == "Title":
            # Flush pending list items before changing section
            if list_buffer:
                chunk = _make_list_chunk(
                    list_items=list_buffer,
                    context=pending_context,
                    document_id=loaded_doc.document_id,
                    section_title=current_section,
                    section_path=_build_path(section_stack),
                    page_hint=element.metadata.get("page_number", 1),
                    chunk_index=chunk_index,
                    doc_title=doc_title,
                )
                chunks.append(chunk)
                chunk_index += 1
                list_buffer = []
                pending_context = None

            # Get heading level from metadata (set by loader's markdown parser)
            # LlamaParse sets heading_level: # = 1, ## = 2, ### = 3
            # Unstructured / PyMuPDF fallback may not set it — default to 1
            heading_level = element.metadata.get("heading_level", 1)

            # Add this heading to the stack
            section_stack[heading_level] = element.text

            # Clear deeper levels — moving from H2 to H1 clears H2, H3, H4
            for level in list(section_stack.keys()):
                if level > heading_level:
                    del section_stack[level]

            current_section = element.text
            continue

        # ── Header/Footer: discard — never useful for retrieval ──
        if el_type in ("Header", "Footer", "PageBreak"):
            continue

        # ── Table: keep whole always ──
        # Catches both explicit Table elements AND markdown tables LlamaParse
        # returned as NarrativeText (safety net via _is_markdown_table)
        if el_type == "Table" or _is_markdown_table(element.text):
            table_text = element.html or element.text
            section_path = _build_path(section_stack)
            chunk_text = f"[{doc_title}] [{section_path}]\n\nTable:\n{table_text}"

            chunks.append(Chunk(
                chunk_id=_make_id(loaded_doc.document_id, chunk_text),
                document_id=loaded_doc.document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                element_type="Table",           # Force Table even if LlamaParse said NarrativeText
                section_title=current_section,
                section_path=section_path,
                page_hint=element.metadata.get("page_number", 1),
                is_table=True,
                token_count=len(chunk_text) // 4,
            ))
            chunk_index += 1
            continue

        # ── ListItem: buffer for grouping with context ──
        if el_type == "ListItem":
            list_buffer.append(element.text)
            continue

        # ── NarrativeText: flush list buffer then process ──
        if el_type == "NarrativeText":
            # Flush pending list items first
            if list_buffer:
                chunk = _make_list_chunk(
                    list_items=list_buffer,
                    context=pending_context,
                    document_id=loaded_doc.document_id,
                    section_title=current_section,
                    section_path=_build_path(section_stack),
                    page_hint=element.metadata.get("page_number", 1),
                    chunk_index=chunk_index,
                    doc_title=doc_title,
                )
                chunks.append(chunk)
                chunk_index += 1
                list_buffer = []
                pending_context = None

            # Save as context for upcoming list items
            pending_context = element.text

            # Chunk the narrative
            new_chunks = _chunk_narrative(
                text=element.text,
                section_title=current_section,
                section_path=_build_path(section_stack),
                document_id=loaded_doc.document_id,
                page_hint=element.metadata.get("page_number", 1),
                chunk_index_start=chunk_index,
                doc_title=doc_title,
            )
            chunks.extend(new_chunks)
            chunk_index += len(new_chunks)

    # Flush any remaining list items at end of document
    if list_buffer:
        chunk = _make_list_chunk(
            list_items=list_buffer,
            context=pending_context,
            document_id=loaded_doc.document_id,
            section_title=current_section,
            section_path=_build_path(section_stack),
            page_hint=1,
            chunk_index=chunk_index,
            doc_title=doc_title,
        )
        chunks.append(chunk)

    return chunks


def _chunk_narrative(
    text: str,
    section_title: str,
    section_path: str,
    document_id: str,
    page_hint: int,
    chunk_index_start: int,
    doc_title: str = "",
) -> List[Chunk]:
    """
    Chunk narrative text.
    Short (< 300 tokens): keep whole.
    Long (> 300 tokens): split by sentence with 2-sentence overlap.
    section_path is prepended to every chunk so the embedding knows
    WHERE in the document this content lives.
    """
    # Prepend doc title + section path — gives dense/sparse models full context
    # "[EMPLOYEE HANDBOOK] [Employee Classification Policy]\n\nEmployees are..."
    # is far more retrievable than just "Employees are classified as..."
    context_prefix = f"[{doc_title}] [{section_path}]\n\n" if doc_title else f"{section_path}\n\n"
    token_count = len(text) // 4

    # Short section — keep whole
    if token_count <= SHORT_THRESHOLD:
        chunk_text = context_prefix + text
        return [Chunk(
            chunk_id=_make_id(document_id, chunk_text),
            document_id=document_id,
            text=chunk_text,
            chunk_index=chunk_index_start,
            element_type="NarrativeText",
            section_title=section_title,
            section_path=section_path,
            page_hint=page_hint,
            is_table=False,
            token_count=len(chunk_text) // 4,
        )]

    # Long section — split by sentence with overlap
    # Split on sentence boundaries while preserving punctuation
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []
    current_tokens = 0
    idx = chunk_index_start
    prefix_tokens = len(context_prefix) // 4

    for sentence in sentences:
        s_tokens = len(sentence) // 4

        if current_tokens + s_tokens > MAX_NARRATIVE_TOKENS - prefix_tokens and current:
            chunk_text = context_prefix + " ".join(current)
            chunks.append(Chunk(
                chunk_id=_make_id(document_id, chunk_text),
                document_id=document_id,
                text=chunk_text,
                chunk_index=idx,
                element_type="NarrativeText",
                section_title=section_title,
                section_path=section_path,
                page_hint=page_hint,
                is_table=False,
                token_count=len(chunk_text) // 4,
            ))
            idx += 1

            # Keep last 2 sentences as overlap for reading continuity
            current = current[-OVERLAP_SENTENCES:] if len(current) >= OVERLAP_SENTENCES else current[:]
            current_tokens = sum(len(s) // 4 for s in current)

        current.append(sentence)
        current_tokens += s_tokens

    # Final chunk
    if current:
        chunk_text = context_prefix + " ".join(current)
        chunks.append(Chunk(
            chunk_id=_make_id(document_id, chunk_text),
            document_id=document_id,
            text=chunk_text,
            chunk_index=idx,
            element_type="NarrativeText",
            section_title=section_title,
            section_path=section_path,
            page_hint=page_hint,
            is_table=False,
            token_count=len(chunk_text) // 4,
        ))

    return chunks


def _make_list_chunk(
    list_items: List[str],
    context: Optional[str],
    document_id: str,
    section_title: str,
    section_path: str,
    page_hint: int,
    chunk_index: int,
    doc_title: str = "",
) -> Chunk:
    """
    Group list items with their preceding context paragraph.
    List without context loses meaning — "• 15 days" means nothing
    without "Full-time employees are entitled to:" before it.
    """
    context_prefix = f"[{doc_title}] [{section_path}]\n\n" if doc_title else f"{section_path}\n\n"
    context_part = f"{context}\n\n" if context else ""
    list_text = "\n".join(f"• {item}" for item in list_items)
    chunk_text = context_prefix + context_part + list_text

    return Chunk(
        chunk_id=_make_id(document_id, chunk_text),
        document_id=document_id,
        text=chunk_text,
        chunk_index=chunk_index,
        element_type="ListItem",
        section_title=section_title,
        section_path=section_path,
        page_hint=page_hint,
        is_table=False,
        token_count=len(chunk_text) // 4,
    )


def _make_id(document_id: str, text: str) -> str:
    """Deterministic chunk ID — same content always produces same ID for deduplication."""
    content = f"{document_id}::{text[:200]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]