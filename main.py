"""FastAPI app that uses Gemini via LangChain to answer questions from local PDFs."""
from __future__ import annotations

import glob
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
# En Python es habitual definir constantes al inicio del archivo. Aquí reunimos
# los parámetros de configuración que controlan cómo se comporta la aplicación.
# Cada uno de ellos se obtiene desde variables de entorno. Esto permite ajustar
# la aplicación (por ejemplo, cambiar la carpeta de PDF o el directorio de
# almacenamiento) sin necesidad de modificar el código fuente.
PDF_FOLDER = os.getenv("PDF_FOLDER", "pdfs")
# Ubicación donde se guardará el índice vectorial que utiliza Chroma para
# realizar búsquedas semánticas rápidas sobre los documentos cargados.
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
# Clave de acceso para el servicio Gemini de Google. Es obligatoria y debe
# proporcionarse fuera del código (por motivos de seguridad) antes de arrancar
# la API.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Must be set externally before running the app.

app = FastAPI(title="PDF Question Answering API")

# ``qa_chain`` se inicializa en ``None`` y posteriormente se rellenará en el
# evento de inicio de FastAPI. Usar una variable global evita reconstruir toda la
# tubería de LangChain en cada petición HTTP, lo que reduciría el rendimiento.
qa_chain: RetrievalQA | None = None


class AskRequest(BaseModel):
    """Schema for incoming POST /ask requests."""

    # Atributo único que representa la pregunta del usuario. Pydantic validará
    # automáticamente que se reciba como cadena de texto en el cuerpo JSON.
    question: str


class AskResponse(BaseModel):
    """Schema for responses returned by POST /ask."""

    # Respuesta generada por el modelo Gemini.
    answer: str
    # Lista de fragmentos de texto provenientes de los PDF utilizados para
    # respaldar la respuesta. Se envía como lista para que el cliente pueda
    # mostrar cada cita por separado.
    context: List[str]


def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """Load PDFs from the provided directory and convert every page into a LangChain Document."""

    # ``documents`` almacenará todos los textos extraídos. Cada elemento será un
    # objeto ``Document`` de LangChain que contiene contenido y metadatos.
    documents: List[Document] = []
    # ``glob`` permite buscar de forma recursiva todos los archivos con
    # extensión .pdf dentro de ``pdf_dir`` y sus subdirectorios.
    pdf_paths = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)

    for path in pdf_paths:
        # ``PdfReader`` abre el archivo PDF para leer sus páginas.
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages):
            # ``extract_text`` devuelve el contenido en formato de texto plano.
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        # ``page_content`` es el texto que se utilizará para el
                        # análisis semántico posterior.
                        page_content=text,
                        # Los metadatos guardan información contextual. Aquí se
                        # conserva la ruta del PDF y el número de página para
                        # poder citar el origen de cada fragmento más adelante.
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

    # Las "embeddings" convierten cada fragmento de texto en un vector de
    # números que captura su significado. Este modelo específico proviene del
    # servicio Gemini.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Split documents into smaller chunks to improve retrieval granularity.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    # ``split_documents`` divide el contenido largo en secciones superpuestas.
    # Este paso mejora la precisión de la búsqueda, porque cada vector
    # representa una idea más concreta.
    split_docs = splitter.split_documents(documents)

    # Create or load the Chroma vector store persisted on disk.
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vector_store


def initialize_qa_chain() -> RetrievalQA:
    """Construct the RetrievalQA chain that uses Gemini for answer generation."""

    # Cargar los documentos es el primer paso para poder construir el índice.
    documents = load_pdf_documents(PDF_FOLDER)
    if not documents:
        raise RuntimeError(
            f"No PDF documents found in '{PDF_FOLDER}'. Add PDFs before starting the service."
        )

    # ``build_vector_store`` devuelve un almacén vectorial listo para ser
    # consultado. Este almacén contiene los vectores calculados para cada
    # fragmento de los PDF.
    vector_store = build_vector_store(documents)

    # ``ChatGoogleGenerativeAI`` es el wrapper de LangChain que permite invocar a
    # Gemini como modelo conversacional. ``temperature`` controla la aleatoriedad
    # en las respuestas (valores bajos = respuestas más deterministas).
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Build a retriever from the vector store to be consumed by LangChain's RetrievalQA chain.
    # Un "retriever" se encarga de buscar los fragmentos más relevantes dentro
    # del índice vectorial dado un texto de consulta.
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # ``RetrievalQA`` combina el retriever con el modelo generativo. Primero se
    # buscan los documentos más cercanos semánticamente y luego se pasan al LLM
    # para construir la respuesta final.
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


@app.on_event("startup")
def on_startup() -> None:
    """FastAPI startup hook to prepare the LangChain pipeline once when the server launches."""

    # FastAPI ejecuta esta función automáticamente al iniciar el servidor.
    # Construir el ``qa_chain`` aquí garantiza que los recursos pesados (lectura
    # de PDF, generación de embeddings, etc.) se realicen una única vez.
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
    # El objeto ``qa_chain`` se comporta como una función: recibe un diccionario
    # con la clave "query" y devuelve otro diccionario que incluye la respuesta
    # generada y los documentos relevantes.
    result = qa_chain({"query": payload.question})
    answer = result.get("result", "No answer generated.")

    source_docs = result.get("source_documents", [])
    # ``context_snippets`` almacenará los textos que se devolverán al cliente
    # como referencia, evitando repeticiones mediante ``seen_sources``.
    context_snippets = []
    seen_sources = set()
    for doc in source_docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown source")
        page = metadata.get("page")

        # Normalize the snippet content so duplicates can be filtered reliably.
        page_content = (doc.page_content or "").strip()

        # Use the page information when available; otherwise fall back to the
        # combination of source and content to avoid losing distinct snippets
        # that originate from the same file but different sections.
        # En términos prácticos: si conocemos el número de página usamos ese dato
        # para evitar duplicados. Si no está disponible, utilizamos el contenido
        # del fragmento como alternativa.
        if page in (None, ""):
            dedupe_key = (source, page_content)
            page_display = "?"
        else:
            dedupe_key = (source, page)
            page_display = page

        # ``seen_sources`` recuerda qué combinaciones ya se han agregado para no
        # devolver el mismo fragmento varias veces al cliente.
        if dedupe_key in seen_sources:
            continue
        seen_sources.add(dedupe_key)

        snippet = f"Source: {source} (page {page_display})\n{page_content}"
        context_snippets.append(snippet)

    return AskResponse(answer=answer, context=context_snippets)


@app.get("/")
def root() -> dict[str, str]:
    """Simple health check endpoint."""

    # Responder con un objeto simple permite a servicios externos verificar que
    # la API está viva sin ejecutar el proceso completo de pregunta/respuesta.
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Permite ejecutar la aplicación directamente con ``python main.py`` durante
    # el desarrollo. Uvicorn es el servidor ASGI recomendado para FastAPI.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
