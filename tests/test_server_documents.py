from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from mfq.server.runtime.backend import BackendDelta
from mfq.server.services.documents import DocumentExtractionError, extract_document
import mfq.server.services.documents as documents
from mfq.server.protocol.models import CreateDocumentRequest, CreateSessionRequest
from mfq.server.services.service import ServerService
from mfq.server.state.storage import SessionStore
from tests.test_server_service import FakeBackend


def _docx_bytes(text: str) -> bytes:
    output = io.BytesIO()
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", document)
    return output.getvalue()


def test_plain_text_and_docx_extraction(tmp_path: Path) -> None:
    text = tmp_path / "notes.md"
    text.write_text("MFQ document text", encoding="utf-8")
    assert extract_document(text, "text/markdown", text.name).text == "MFQ document text"

    docx = tmp_path / "notes.docx"
    docx.write_bytes(_docx_bytes("MFQ DOCX text"))
    extracted = extract_document(
        docx,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        docx.name,
    )
    assert extracted.text == "MFQ DOCX text"
    assert extracted.extractor == "docx-xml-v1"


def test_binary_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"\xff\x00\xfe")
    with pytest.raises(DocumentExtractionError):
        extract_document(path, "text/plain", path.name)


def test_compressed_docx_is_bounded_before_xml_is_loaded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(documents, "MAX_EXTRACTION_BYTES", 1024)
    path = tmp_path / "large.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b" " * 4096)
    assert path.stat().st_size < 1024
    def unexpected(*args, **kwargs):
        pytest.fail("oversized compressed XML must be rejected before decompression")
    monkeypatch.setattr(zipfile.ZipFile, "open", unexpected)
    with pytest.raises(DocumentExtractionError, match="extraction size limit"):
        extract_document(path, "application/octet-stream", path.name)


@pytest.mark.parametrize("error", [RuntimeError("password required"), NotImplementedError("unsupported compression")])
def test_unsupported_docx_archive_is_a_document_error(tmp_path: Path, monkeypatch, error) -> None:
    path = tmp_path / "unsupported.docx"
    path.write_bytes(_docx_bytes("text"))
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(zipfile.ZipFile, "open", fail)
    with pytest.raises(DocumentExtractionError, match="invalid DOCX document"):
        extract_document(path, "application/octet-stream", path.name)


def test_pdf_stops_extracting_pages_at_the_text_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(documents, "MAX_EXTRACTED_CHARACTERS", 10)
    def unexpected():
        pytest.fail("later PDF pages must not be extracted after the text limit")
    reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "x" * 11),
        SimpleNamespace(extract_text=unexpected)])
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _: reader))
    with pytest.raises(DocumentExtractionError, match="extracted text exceeds"):
        extract_document(tmp_path / "large.pdf", "application/pdf", "large.pdf")


def test_text_read_is_bounded_before_decoding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(documents, "MAX_EXTRACTION_BYTES", 1024)
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 1025)
    with pytest.raises(DocumentExtractionError, match="extraction size limit"):
        extract_document(path, "text/plain", path.name)


def test_document_is_persisted_and_expanded_for_backend(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = FakeBackend((BackendDelta(content_delta="ok", finish_reason="stop"),))
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, backend)
        data = b"A persistent document"
        media = await service.upload_media(data, "text/plain", hashlib.sha256(data).hexdigest())
        document = await service.create_document(
            CreateDocumentRequest(media_id=media.media.id, name="notes.txt")
        )
        assert document.text == "A persistent document"
        assert (await service.get_document(media.media.id)).extractor == "plain-text-v1"

        session = await service.create_session(CreateSessionRequest(model="model-a"))
        from uuid import uuid4

        from mfq.server.protocol.models import CreateResponseRequest

        prepared = await service.prepare_response(
            session.id,
            CreateResponseRequest(
                request_id=uuid4(),
                expected_revision=0,
                input=[{"type": "document", "media": media.media, "name": "notes.txt"}],
            ),
        )
        assert prepared.backend_messages[-1]["content"].startswith('<document name="notes.txt">')
        await service.collect_response(prepared)

    asyncio.run(scenario())
