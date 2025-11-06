"""FastAPI app that uses Gemini via LangChain to answer questions from local PDFs."""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Configuration constants controlled by environment variables for flexibility.
# Carga las variables del archivo .env
load_dotenv()#
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

# Diccionario en memoria que conserva el historial de cada combinación
# usuario/conversación. La clave es una tupla ``(user_id, conversation_id)`` y
# el valor es una lista de pares ``(pregunta, respuesta)`` en orden cronológico.
ConversationHistory = List[Tuple[str, str]]
conversation_histories: Dict[Tuple[str, str], ConversationHistory] = {}

# Número máximo de turnos previos que se incluirán al condensar el historial
# de chat. Mantenerlo acotado evita que el prompt crezca sin control mientras
# conserva suficiente contexto para referencias recientes.
MAX_HISTORY_TURNS = 3


class AskRequest(BaseModel):
    """Schema for incoming POST /ask requests."""

    # Atributo único que representa la pregunta del usuario. Pydantic validará
    # automáticamente que se reciba como cadena de texto en el cuerpo JSON.
    question: str
    # Identificador obligatorio del usuario que realiza la pregunta. Permite
    # mantener historiales independientes por cada persona que interactúa con
    # la API.
    user_id: str
    # Identificador opcional de conversación. Si no se envía, todas las
    # preguntas de un usuario compartirán el mismo historial.
    conversation_id: str | None = None


class AskResponse(BaseModel):
    """Schema for responses returned by POST /ask."""

    # Respuesta generada por el modelo Gemini.
    answer: str
    # Lista de fragmentos de texto provenientes de los PDF utilizados para
    # respaldar la respuesta. Se envía como lista para que el cliente pueda
    # mostrar cada cita por separado.
    context: List[str]
    # Identificador de conversación efectivo que se utilizó para almacenar el
    # historial. El cliente puede conservarlo para futuras solicitudes.
    conversation_id: str


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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # Prompt específico para guiar al modelo en el formato requerido por el
    # frontend. El ``PromptTemplate`` se integra con LangChain y recibe el
    # contexto recuperado y la pregunta original.
    prompt = PromptTemplate(
        template=(
            """📋 Instrucciones:
            - Usa únicamente HTML básico: <h2>, <p>, <ul>, <li>, <strong>.
            - No incluyas estilos en línea ni enlaces.
            - Estructura tus respuestas así:
            1. Un <p> inicial con la explicación principal (máximo 3 líneas).
            2. Si aplica, una lista <ul><li> con puntos o pasos clave.
            3. Al final, una breve referencia a la fuente si está disponible.

            ⚠️ Reglas de comportamiento:
            - Si el usuario solo saluda (“hola”, “buen día”, “gracias”, etc.), responde amablemente con un saludo breve y servicial.**no incluyas ningún contenido del contexto**.
            - Si la pregunta **no tiene relación con el contexto** o el contexto **no contiene información útil**, responde exactamente:
            <p>No encontré información relacionada en la documentación.</p>
            - Si la respuesta requiere información extensa, **resume solo lo esencial** (máximo 5 líneas de texto total).
            - No cites todo el documento ni fragmentos largos.
            - No uses frases como “según la información proporcionada” ni “de acuerdo al contexto”.

            Historial reciente de la conversación (usuario → asistente):
            {chat_history_text}

            "Contexto disponible:\n{context}\n\n"
            "Pregunta del usuario: {question}\n\n"
            "Respuesta en HTML:"""
        ),
        input_variables=["context", "question", "chat_history_text"],
    )

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
        chain_type_kwargs={"prompt": prompt},
    )


def summarize_chat_history(chat_history: ConversationHistory, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Condense the chat history into a short textual summary for the prompt."""

    if not chat_history:
        return "(sin historial previo)"

    relevant_turns = chat_history[-max_turns:]
    lines: List[str] = []
    for idx, (question, answer) in enumerate(relevant_turns, start=1):
        lines.append(f"Turno {idx} - Usuario: {question}")
        lines.append(f"Turno {idx} - Asistente: {answer}")
    return "\n".join(lines)


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

    if not payload.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    # Normaliza el identificador de conversación. Si el cliente no especifica
    # uno, se utiliza "default" para agrupar todas las preguntas de ese usuario.
    conversation_id = payload.conversation_id or "default"
    session_key = (payload.user_id, conversation_id)
    history = conversation_histories.setdefault(session_key, [])

    # Se envía el historial completo a la cadena de LangChain. Algunos modelos
    # lo ignoran, pero otros pueden aprovecharlo para respuestas más coherentes.
    chat_history = list(history)

    # Run the retrieval QA chain to obtain both the answer and the source documents.
    # El objeto ``qa_chain`` se comporta como una función: recibe un diccionario
    # con la clave "query" y devuelve otro diccionario que incluye la respuesta
    # generada y los documentos relevantes.
    chat_history_text = summarize_chat_history(chat_history)

    result = qa_chain(
        {
            "query": payload.question,
            "chat_history": chat_history,
            "chat_history_text": chat_history_text,
        }
    )
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

    # Actualiza el historial almacenando la nueva pareja pregunta/respuesta.
    history.append((payload.question, answer))

    return AskResponse(
        answer=answer,
        context=context_snippets,
        conversation_id=conversation_id,
    )


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
