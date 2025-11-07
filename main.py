"""FastAPI app that uses Gemini via LangChain to answer questions from local PDFs."""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import List, Protocol, Tuple

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
from redis import Redis
from redis.exceptions import RedisError

# Configuration constants controlled by environment variables for flexibility.
# Carga las variables del archivo .env
load_dotenv()#
# Configuramos un logger de módulo para reportar problemas con el backend de
# historiales sin inundar stdout.
logger = logging.getLogger(__name__)


def _read_positive_int(name: str, *, default: int | None = None) -> int | None:
    """Leer una variable de entorno y devolverla como entero positivo."""

    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        parsed = int(raw_value)
    except ValueError as exc:  # pragma: no cover - solo se activa ante mala configuración.
        raise RuntimeError(f"{name} debe ser un entero válido.") from exc

    if parsed <= 0:
        return None
    return parsed
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
# URL de conexión a Redis que compartirán todos los workers. Se puede ajustar a
# servicios administrados (por ejemplo, ElastiCache) sin modificar el código:
# en producción basta con definir la variable de entorno ``REDIS_URL`` con la
# cadena de conexión adecuada.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Máximo de turnos que se conservarán por conversación en el backend. No tiene
# por qué coincidir con ``MAX_HISTORY_TURNS`` porque aquí controlamos cuánto
# historial completo queremos persistir.
MAX_STORED_TURNS = _read_positive_int("MAX_STORED_TURNS", default=50)
# Tiempo de vida opcional para cada historial. Si es ``None`` el historial se
# conserva indefinidamente hasta que el usuario lo elimine manualmente.
HISTORY_TTL_SECONDS = _read_positive_int("HISTORY_TTL_SECONDS")

# Alias de tipos para mantener legibles las anotaciones que utilizan listas de
# pares ``(pregunta, respuesta)``.
ConversationHistory = List[Tuple[str, str]]

app = FastAPI(title="PDF Question Answering API")

# ``qa_chain`` se inicializa en ``None`` y posteriormente se rellenará en el
# evento de inicio de FastAPI. Usar una variable global evita reconstruir toda la
# tubería de LangChain en cada petición HTTP, lo que reduciría el rendimiento.
qa_chain: RetrievalQA | None = None

# ``conversation_store`` proporciona un backend compartido (Redis en este caso)
# donde se almacena de forma segura el historial de preguntas/respuestas por
# usuario y conversación. Esto permite que múltiples instancias del servidor
# compartan contexto sin depender de memoria local.
conversation_store: "ConversationStore" | None = None


class ConversationStore(Protocol):
    """Contrato mínimo que deben cumplir los backends de historial."""

    def load_history(self, user_id: str, conversation_id: str) -> ConversationHistory:
        ...

    def append_turn(
        self, user_id: str, conversation_id: str, question: str, answer: str
    ) -> None:
        ...


class ConversationStoreError(RuntimeError):
    """Error controlado que indica fallos al acceder al backend de historiales."""


class RedisConversationStore:
    """Implementación basada en listas Redis para almacenar turnos de conversación."""

    def __init__(
        self,
        client: Redis,
        *,
        max_length: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._client = client
        self._max_length = max_length
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _build_key(user_id: str, conversation_id: str) -> str:
        # Los dos identificadores se concatenan de forma directa. Redis permite
        # caracteres especiales, pero usar ``:`` facilita inspeccionar claves.
        return f"chat:{user_id}:{conversation_id}"

    def load_history(self, user_id: str, conversation_id: str) -> ConversationHistory:
        key = self._build_key(user_id, conversation_id)
        try:
            entries = self._client.lrange(key, 0, -1)
        except RedisError as exc:  # pragma: no cover - solo ante fallos de red.
            raise ConversationStoreError("No fue posible recuperar el historial desde Redis.") from exc

        history: ConversationHistory = []
        for raw_entry in entries:
            try:
                payload = json.loads(raw_entry)
            except json.JSONDecodeError:
                # Si encontramos datos corruptos los ignoramos para no bloquear la conversación.
                logger.warning("Entrada de historial dañada en Redis para %s", key)
                continue

            question = payload.get("q", "")
            answer = payload.get("a", "")
            history.append((question, answer))
        return history

    def append_turn(
        self, user_id: str, conversation_id: str, question: str, answer: str
    ) -> None:
        key = self._build_key(user_id, conversation_id)
        entry = json.dumps({"q": question, "a": answer}, ensure_ascii=False)
        try:
            pipeline = self._client.pipeline()
            pipeline.rpush(key, entry)
            if self._max_length:
                pipeline.ltrim(key, -self._max_length, -1)
            if self._ttl_seconds:
                pipeline.expire(key, self._ttl_seconds)
            pipeline.execute()
        except RedisError as exc:  # pragma: no cover - solo ante fallos de red.
            raise ConversationStoreError("No fue posible guardar el historial en Redis.") from exc


class InMemoryConversationStore:
    """Versión en memoria útil para pruebas unitarias o desarrollo local."""

    def __init__(self, *, max_length: int | None = None) -> None:
        self._storage: dict[str, ConversationHistory] = {}
        self._max_length = max_length

    @staticmethod
    def _build_key(user_id: str, conversation_id: str) -> str:
        return f"{user_id}::{conversation_id}"

    def load_history(self, user_id: str, conversation_id: str) -> ConversationHistory:
        key = self._build_key(user_id, conversation_id)
        history = self._storage.get(key, [])
        # Devolvemos una copia para evitar que el llamador modifique el listado interno.
        return list(history)

    def append_turn(
        self, user_id: str, conversation_id: str, question: str, answer: str
    ) -> None:
        key = self._build_key(user_id, conversation_id)
        history = self._storage.setdefault(key, [])
        history.append((question, answer))
        if self._max_length and len(history) > self._max_length:
            self._storage[key] = history[-self._max_length :]

    def clear(self) -> None:
        self._storage.clear()


def create_conversation_store() -> ConversationStore:
    """Inicializa el backend de historiales compartido usando Redis."""

    try:
        client = Redis.from_url(REDIS_URL, decode_responses=True)
        # ``ping`` fuerza una conexión temprana y falla rápido si Redis no está disponible.
        client.ping()
    except RedisError as exc:  # pragma: no cover - se activa solo si Redis está caído.
        raise ConversationStoreError("No se pudo conectar a Redis para el historial de chats.") from exc

    return RedisConversationStore(
        client,
        max_length=MAX_STORED_TURNS,
        ttl_seconds=HISTORY_TTL_SECONDS,
    )

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


def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Return the embeddings model configured for the application."""

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable must be set with a valid Gemini API key."
        )

    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def build_vector_store(
    documents: List[Document],
    persist_directory: str | None = None,
) -> Chroma:
    """Create (or update) a Chroma vector store with Google embeddings from the documents."""

    embeddings = _get_embeddings_model()

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

    directory = persist_directory or CHROMA_DIR
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=directory,
    )
    vector_store.persist()
    return vector_store


def load_vector_store(persist_directory: str | None = None) -> Chroma:
    """Open the existing Chroma store previously generated offline."""

    embeddings = _get_embeddings_model()

    directory = Path(persist_directory or CHROMA_DIR)
    if not directory.exists() or not any(directory.iterdir()):
        raise RuntimeError(
            f"Chroma directory '{directory}' is missing or empty. Run the offline indexing pipeline first."
        )

    return Chroma(persist_directory=str(directory), embedding_function=embeddings)


def initialize_qa_chain() -> RetrievalQA:
    """Construct the RetrievalQA chain that uses Gemini for answer generation."""

    # ``load_vector_store`` abre el índice persistente generado previamente.
    vector_store = load_vector_store()


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
    global qa_chain, conversation_store
    conversation_store = create_conversation_store()
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
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store is not ready yet.")

    try:
        # Se recupera el historial desde Redis para que todos los workers
        # compartan exactamente el mismo contexto.
        chat_history = conversation_store.load_history(payload.user_id, conversation_id)
    except ConversationStoreError as exc:
        logger.error("Fallo al recuperar el historial: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to load conversation history.") from exc

    # Run the retrieval QA chain to obtain both the answer and the source documents.
    # El objeto ``qa_chain`` se comporta como una función: recibe un diccionario
    # con la clave "query" y devuelve otro diccionario que incluye la respuesta
    # generada y los documentos relevantes.
    chat_history_text = summarize_chat_history(chat_history)

    # ``RetrievalQA`` no reconoce variables adicionales en cada llamada. Para
    # poder incluir el historial resumido en el prompt personalizado realizamos
    # un ``partial`` del template antes de ejecutar la cadena y restauramos el
    # original al finalizar. De esta forma "chat_history_text" queda fijado en
    # el prompt sin que LangChain exija recibirlo explícitamente como parámetro.
    combine_chain = getattr(qa_chain, "combine_documents_chain", None)
    llm_chain = getattr(combine_chain, "llm_chain", None)
    original_prompt = getattr(llm_chain, "prompt", None)

    if llm_chain is not None and original_prompt is not None:
        llm_chain.prompt = original_prompt.partial(chat_history_text=chat_history_text)

    print(chat_history_text)

    try:
        result = qa_chain(
            {
                "query": payload.question,
                "chat_history": chat_history,
                "chat_history_text": chat_history_text,
            }
        )
    finally:
        if llm_chain is not None and original_prompt is not None:
            llm_chain.prompt = original_prompt
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

    # Actualiza el historial almacenando la nueva pareja pregunta/respuesta en
    # el backend compartido. ``append_turn`` usa operaciones atómicas (RPUSH +
    # LTRIM) para evitar condiciones de carrera entre múltiples peticiones.
    try:
        conversation_store.append_turn(payload.user_id, conversation_id, payload.question, answer)
    except ConversationStoreError as exc:
        logger.error("Fallo al guardar el historial: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to persist conversation history.") from exc

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
