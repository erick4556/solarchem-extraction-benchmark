"""RDF export of the SolarChem *silver* ground truth.

Gold is not an input: it was only used to pick the OCR engine. The graph is
built from the LightOn working silver (``ground_truth_lighton_ocr_302.json``).

Instances use the ``sc:`` alignment ontology (Article, TableRepresentation,
TextMention). Cell grids stay in JSON; RDF records identity, caption, section
and in-text mentions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from solarchem_benchmark.gt.context import caption_title
from solarchem_benchmark.gt.schema import GroundTruthDocument, Table

SC = "https://w3id.org/solarchem/alignment/"
INST = "https://w3id.org/solarchem/benchmark/"

_PREFIXES = """\
@prefix sc:      <https://w3id.org/solarchem/alignment/> .
@prefix inst:    <https://w3id.org/solarchem/benchmark/> .
@prefix schema:  <https://schema.org/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
@prefix csvw:    <http://www.w3.org/ns/csvw#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
"""

_LOCAL_RE = re.compile(r"[^A-Za-z0-9_-]+")


def is_gold_path(path: Path) -> bool:
    """True when ``path`` sits under the Gold-pilot directory."""
    return any(part.lower() == "gold" for part in path.parts)


def local_name(value: str) -> str:
    """Turn a document or table id into a Turtle local name."""
    slug = _LOCAL_RE.sub("_", value).strip("_")
    if not slug or slug[0].isdigit():
        slug = f"n_{slug}"
    return slug


def ttl_literal(value: str) -> str:
    """Escape ``value`` as a Turtle string."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _curie(kind: str, ident: str) -> str:
    return f"inst:{kind}_{local_name(ident)}"


def article_iri(document: GroundTruthDocument) -> str:
    return _curie("article", document.document_id)


def table_iri(table: Table) -> str:
    return _curie("table", table.table_id)


def mention_iri(table: Table, index: int) -> str:
    return f"{table_iri(table)}_mention_{index}"


def document_to_turtle(document: GroundTruthDocument) -> str:
    """Serialise one silver document (article + tables + mentions)."""
    chunks: list[str] = []
    article = article_iri(document)
    statements = [f"{article} a sc:Article"]
    if document.title.strip():
        statements.append(f"    schema:name {ttl_literal(document.title.strip())}")
    pdf = document.source_pdf.strip()
    if pdf:
        statements.append(f"    schema:identifier {ttl_literal(pdf)}")
        statements.append(f"    prov:hadPrimarySource {ttl_literal(pdf)}")
    chunks.append(" ;\n".join(statements) + " .\n")

    for table in document.tables:
        chunks.append(_table_block(article, table))
    return "\n".join(chunks)


def _table_block(article: str, table: Table) -> str:
    iri = table_iri(table)
    name = table.table_label.strip() or table.table_id
    title = caption_title(table.caption).strip()
    if title:
        name = f"{table.table_label.strip() or 'Table'} — {title}" if table.table_label.strip() else title

    statements = [
        f"{iri} a sc:TableRepresentation",
        f"    schema:name {ttl_literal(name)}",
        f"    sc:belongsToArticle {article}",
        f"    schema:pageStart {int(table.page)}",
    ]
    if table.caption.strip():
        statements.append(f"    schema:description {ttl_literal(table.caption.strip())}")
    section = table.context.section_title.strip()
    if section:
        statements.append(f"    schema:articleSection {ttl_literal(section)}")
    for column in table.columns:
        if column.strip():
            statements.append(f"    csvw:column [ csvw:name {ttl_literal(column.strip())} ]")
    blocks = [" ;\n".join(statements) + " .\n"]

    for index, mention in enumerate(table.context.mentions, start=1):
        text = mention.text.strip()
        if not text:
            continue
        mention_id = mention_iri(table, index)
        mention_statements = [
            f"{mention_id} a sc:TextMention",
            f"    schema:text {ttl_literal(text)}",
            f"    schema:about {iri}",
            f"    prov:wasDerivedFrom {article}",
            f"    schema:pageStart {int(mention.page)}",
        ]
        blocks.append(" ;\n".join(mention_statements) + " .\n")
    return "\n".join(blocks)


def corpus_to_turtle(
    documents: Iterable[GroundTruthDocument],
    *,
    source_label: str = "ground_truth_lighton_ocr_302.json",
) -> str:
    """Silver corpus → Turtle instances (ontology is a separate file)."""
    docs = list(documents)
    header = (
        "# Silver LightOn table+context instances. Load together with\n"
        "# solarchem-alignment.ttl (sc: schema). Not derived from Gold.\n"
        f"# Source: {source_label}\n\n"
        + _PREFIXES
        + "\n"
    )
    body = "\n".join(document_to_turtle(document) for document in docs)
    return header + body


def ontology_path() -> Path:
    """Packaged ``sc:`` schema (classes and shared vocabulary, no paper instances)."""
    return Path(__file__).resolve().parent / "solarchem-alignment.ttl"
