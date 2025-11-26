import sys
from types import ModuleType
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_stub_modules() -> None:
    """Install lightweight stand-ins for heavy dependencies used in main.py."""

    # langchain_classic.chains
    langchain_classic = ModuleType("langchain_classic")
    langchain_classic_chains = ModuleType("langchain_classic.chains")

    class _RetrievalQA:
        @classmethod
        def from_chain_type(cls, *args, **kwargs):  # pragma: no cover - not expected to run
            return cls()

    langchain_classic_chains.RetrievalQA = _RetrievalQA
    sys.modules["langchain_classic"] = langchain_classic
    sys.modules["langchain_classic.chains"] = langchain_classic_chains

    # langchain_core.documents
    langchain_core = ModuleType("langchain_core")
    langchain_core_documents = ModuleType("langchain_core.documents")
    langchain_core_prompts = ModuleType("langchain_core.prompts")

    class _Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    langchain_core_documents.Document = _Document
    class _PromptTemplate:
        def __init__(self, template="", input_variables=None):
            self.template = template
            self.input_variables = input_variables or []

    langchain_core_prompts.PromptTemplate = _PromptTemplate

    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.documents"] = langchain_core_documents
    sys.modules["langchain_core.prompts"] = langchain_core_prompts

    # langchain_text_splitters
    langchain_text_splitters = ModuleType("langchain_text_splitters")

    class _RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, documents):  # pragma: no cover - not expected to run
            return documents

    langchain_text_splitters.RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = langchain_text_splitters

    # langchain_community.vectorstores
    langchain_community = ModuleType("langchain_community")
    langchain_community_vectorstores = ModuleType("langchain_community.vectorstores")
    langchain_community_vectorstores_utils = ModuleType("langchain_community.vectorstores.utils")

    class _Chroma:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_documents(cls, *args, **kwargs):  # pragma: no cover - not expected to run
            return cls()

        def as_retriever(self, *args, **kwargs):  # pragma: no cover - not expected to run
            return self

    langchain_community_vectorstores.Chroma = _Chroma

    def _filter_complex_metadata(documents):
        return documents

    langchain_community_vectorstores_utils.filter_complex_metadata = _filter_complex_metadata

    sys.modules["langchain_community"] = langchain_community
    sys.modules["langchain_community.vectorstores"] = langchain_community_vectorstores
    sys.modules["langchain_community.vectorstores.utils"] = langchain_community_vectorstores_utils

    # langchain_google_genai
    langchain_google_genai = ModuleType("langchain_google_genai")

    class _ChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            pass

    class _GoogleGenerativeAIEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

    langchain_google_genai.ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI
    langchain_google_genai.GoogleGenerativeAIEmbeddings = _GoogleGenerativeAIEmbeddings
    sys.modules["langchain_google_genai"] = langchain_google_genai

    # PyPDF2
    pypdf2 = ModuleType("PyPDF2")

    class _PdfReader:
        def __init__(self, *args, **kwargs):  # pragma: no cover - not expected to run
            self.pages = []

    pypdf2.PdfReader = _PdfReader
    sys.modules["PyPDF2"] = pypdf2

    # redis
    redis_module = ModuleType("redis")

    class _FakeRedis:
        @classmethod
        def from_url(cls, *args, **kwargs):  # pragma: no cover - no se usa en pruebas.
            return cls()

        def ping(self):  # pragma: no cover - no se usa en pruebas.
            pass

    redis_module.Redis = _FakeRedis
    redis_exceptions = ModuleType("redis.exceptions")

    class _RedisError(Exception):
        pass

    redis_exceptions.RedisError = _RedisError
    sys.modules["redis"] = redis_module
    sys.modules["redis.exceptions"] = redis_exceptions

    # prometheus_client
    prometheus_client = ModuleType("prometheus_client")

    class _Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    class _Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

    class _Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    def _generate_latest():
        return b""

    prometheus_client.Histogram = _Histogram
    prometheus_client.Counter = _Counter
    prometheus_client.Gauge = _Gauge
    prometheus_client.generate_latest = _generate_latest
    prometheus_client.CONTENT_TYPE_LATEST = "text/plain"
    sys.modules["prometheus_client"] = prometheus_client


_install_stub_modules()

import main


class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class DummyChain:
    def __init__(self):
        self.calls = []
        self.handler = None

    def __call__(self, query):
        self.calls.append(query)
        if self.handler is not None:
            return self.handler(query)
        return self.default_result("dummy answer")

    @staticmethod
    def default_result(answer):
        return {
            "result": answer,
            "source_documents": [
                DummyDoc("content A", {"source": "file1.pdf", "page": 1}),
                DummyDoc("content A", {"source": "file1.pdf", "page": 1}),
                DummyDoc("content B", {"source": "file2.pdf", "page": 2}),
            ],
        }


@pytest.fixture
def test_client(monkeypatch):
    chain = DummyChain()
    monkeypatch.setattr(main, "initialize_qa_chain", lambda: chain)
    main._dummy_chain = chain  # type: ignore[attr-defined]

    store = main.InMemoryConversationStore(max_length=main.MAX_STORED_TURNS)
    monkeypatch.setattr(main, "create_conversation_store", lambda: store)
    monkeypatch.setattr(main, "create_response_cache", lambda: None)
    main._dummy_store = store  # type: ignore[attr-defined]

    with TestClient(main.app) as client:
        yield client

    main.qa_chain = None
    if hasattr(main, "_dummy_store"):
        main._dummy_store.clear()  # type: ignore[attr-defined]
    main.conversation_store = None
    main.response_cache = None
    main.DOCUMENT_SOURCE_INDEX = {}


def test_ask_question_deduplicates_sources(test_client):
    response = test_client.post("/ask", json={"question": "What?", "user_id": "u1"})
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "dummy answer"
    assert data["context"] == [
        "Source: file1.pdf (page 1)\ncontent A",
        "Source: file2.pdf (page 2)\ncontent B",
    ]
    assert data["conversation_id"] == "default"


def test_conversation_history_is_tracked_per_user(test_client):
    response1 = test_client.post(
        "/ask",
        json={"question": "First?", "user_id": "u1", "conversation_id": "thread"},
    )
    assert response1.status_code == 200

    response2 = test_client.post(
        "/ask",
        json={"question": "Second?", "user_id": "u1", "conversation_id": "thread"},
    )
    assert response2.status_code == 200

    response3 = test_client.post(
        "/ask",
        json={"question": "Other?", "user_id": "u2", "conversation_id": "thread"},
    )
    assert response3.status_code == 200

    calls = main._dummy_chain.calls  # type: ignore[attr-defined]
    assert calls[0]["chat_history"] == []
    assert calls[0]["chat_history_text"] == "(sin historial previo)"
    assert calls[1]["chat_history"] == [("First?", "dummy answer")]
    assert "Turno 1 - Usuario: First?" in calls[1]["chat_history_text"]
    # A different usuario must not receive the previous history.
    assert calls[2]["chat_history"] == []
    assert calls[2]["chat_history_text"] == "(sin historial previo)"


def test_follow_up_question_uses_chat_history(test_client):
    chain = main._dummy_chain  # type: ignore[attr-defined]

    def handler(query):
        if query["query"] == "¿De qué color es el cielo?":
            return chain.default_result("El cielo es azul.")

        assert query["query"] == "¿De qué color es?"
        assert "El cielo es azul." in query["chat_history_text"]
        return chain.default_result("Es azul.")

    chain.handler = handler

    response1 = test_client.post(
        "/ask",
        json={"question": "¿De qué color es el cielo?", "user_id": "u1", "conversation_id": "hist"},
    )
    assert response1.status_code == 200

    response2 = test_client.post(
        "/ask",
        json={"question": "¿De qué color es?", "user_id": "u1", "conversation_id": "hist"},
    )
    assert response2.status_code == 200

    assert response2.json()["answer"] == "Es azul."

    chain.handler = None


def test_ask_includes_image_references(test_client):
    chain = main._dummy_chain  # type: ignore[attr-defined]

    def handler(query):
        return {
            "result": "respuesta",
            "source_documents": [
                DummyDoc(
                    "contenido",
                    {
                        "source": "file1.pdf",
                        "page": 1,
                        "images": [
                            {"src": "https://docs.halconet.com/img/uno.png", "alt": "primera"},
                            {"src": "https://docs.halconet.com/img/uno.png", "alt": "duplicada"},
                        ],
                    },
                )
            ],
        }

    chain.handler = handler

    response = test_client.post("/ask", json={"question": "Imagen?", "user_id": "u1"})
    assert response.status_code == 200

    data = response.json()
    assert data["context"] == ["Source: file1.pdf (page 1)\ncontenido"]

    chain.handler = None


class _DummyResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class _DummySession:
    def __init__(self, sitemap_payload: str, page_payload: str, status_code: int = 200):
        self._sitemap_payload = sitemap_payload
        self._page_payload = page_payload
        self._status_code = status_code
        self.calls = []

    def get(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        if url.endswith("sitemap.xml"):
            return _DummyResponse(self._sitemap_payload, status_code=self._status_code)
        return _DummyResponse(self._page_payload, status_code=self._status_code)


def test_load_halconet_documents_parses_pages():
    sitemap = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
        "<url><loc>https://docs.halconet.com/intro</loc></url>"
        "</urlset>"
    )
    html = (
        "<html><body><h1>Bienvenida</h1>"
        "<p>Contenido principal.</p>"
        "<img src='/assets/fig.png' alt='Diagrama'/>"
        "</body></html>"
    )
    session = _DummySession(sitemap, html)

    documents = main.load_halconet_documents(base_url="https://docs.halconet.com", session=session)

    assert len(documents) == 1
    doc = documents[0]
    assert doc.metadata["source"] == "https://docs.halconet.com/intro"
    assert "Contenido principal" in doc.page_content
    assert doc.metadata["section"] in {"Bienvenida", "Intro"}
    assert "images" not in doc.metadata


def test_load_halconet_documents_keeps_image_only_pages():
    sitemap = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
        "<url><loc>https://docs.halconet.com/manuales</loc></url>"
        "</urlset>"
    )
    html = (
        "<html><body>"
        "<img src='https://docs.halconet.com/wp-content/uploads/2025/11/grupo-tractozone-logo-transparente-e1762967904237.png' alt='Manuales Grupo Tractozone'/>"
        "</body></html>"
    )
    session = _DummySession(sitemap, html)

    documents = main.load_halconet_documents(base_url="https://docs.halconet.com", session=session)

    assert len(documents) == 1
    doc = documents[0]
    assert doc.page_content != ""
    assert "images" not in doc.metadata


def test_load_halconet_documents_handles_errors():
    class _FailingSession:
        def get(self, url: str, timeout: float):  # pragma: no cover - simple stub
            raise requests.RequestException("boom")

    documents = main.load_halconet_documents(base_url="https://docs.halconet.com", session=_FailingSession())
    assert documents == []
