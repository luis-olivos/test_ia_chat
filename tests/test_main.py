from pathlib import Path
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

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

    class _Chroma:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_documents(cls, *args, **kwargs):  # pragma: no cover - not expected to run
            return cls()

        def as_retriever(self, *args, **kwargs):  # pragma: no cover - not expected to run
            return self

    langchain_community_vectorstores.Chroma = _Chroma
    sys.modules["langchain_community"] = langchain_community
    sys.modules["langchain_community.vectorstores"] = langchain_community_vectorstores

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


_install_stub_modules()

import main


class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class DummyChain:
    def __init__(self):
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        return {
            "result": "dummy answer",
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

    with TestClient(main.app) as client:
        yield client

    main.qa_chain = None
    main.conversation_histories.clear()


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
    assert calls[1]["chat_history"] == [("First?", "dummy answer")]
    # A different usuario must not receive the previous history.
    assert calls[2]["chat_history"] == []
