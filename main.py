"""FastAPI app that uses Gemini via LangChain to answer questions from local PDFs."""
from __future__ import annotations

import glob
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader

# Configuration constants controlled by environment variables for flexibility.
PDF_FOLDER = os.getenv("PDF_FOLDER", "pdfs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Must be set externally before running the app.

app = FastAPI(title="PDF Question Answering API")

# Globals that will hold the LangChain objects once initialized during startup.
qa_chain: RetrievalQA | None = None


class AskRequest(BaseModel):
    """Schema for incoming POST /ask requests."""

    question: str


class AskResponse(BaseModel):
    """Schema for responses returned by POST /ask."""

    answer: str
    context: List[str]


def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """Load PDFs from the provided directory and convert every page into a LangChain Document."""

    documents: List[Document] = []
    pdf_paths = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)

    for path in pdf_paths:
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path, "page": page_number + 1},
                    )
                )
    return documents


def build_vector_store(documents: List[Document]) -> Chroma:
    """Create (or update) a Chroma vector store with Google embeddings from the documents."""

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable must be set with a valid Gemini API key."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Split documents into smaller chunks to improve retrieval granularity.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    split_docs = splitter.split_documents(documents)

    # Create or load the Chroma vector store persisted on disk.
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_store.persist()
    return vector_store


def initialize_qa_chain() -> RetrievalQA:
    """Construct the RetrievalQA chain that uses Gemini for answer generation."""

    documents = load_pdf_documents(PDF_FOLDER)
    if not documents:
        raise RuntimeError(
            f"No PDF documents found in '{PDF_FOLDER}'. Add PDFs before starting the service."
        )

    vector_store = build_vector_store(documents)

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Build a retriever from the vector store to be consumed by LangChain's RetrievalQA chain.
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


@app.on_event("startup")
def on_startup() -> None:
    """FastAPI startup hook to prepare the LangChain pipeline once when the server launches."""

    global qa_chain
    qa_chain = initialize_qa_chain()


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """Answer a user's question using Gemini and return the supporting PDF snippets."""

    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA system is not ready yet.")

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Run the retrieval QA chain to obtain both the answer and the source documents.
    result = qa_chain({"query": payload.question})
    answer = result.get("result", "No answer generated.")

    source_docs = result.get("source_documents", [])
    context_snippets = []
    for doc in source_docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", "?")
        snippet = f"Source: {source} (page {page})\n{doc.page_content.strip()}"
        context_snippets.append(snippet)

    return AskResponse(answer=answer, context=context_snippets)


@app.get("/")
def root() -> dict[str, str]:
    """Simple health check endpoint."""

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
