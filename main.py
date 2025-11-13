"""FastAPI app that uses Gemini via LangChain to answer questions from local PDFs."""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, List, Protocol, Tuple, TypeVar

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
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


def _read_non_negative_float(name: str, *, default: float = 0.0) -> float:
    """Leer una variable de entorno y devolverla como ``float`` no negativo."""

    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        parsed = float(raw_value)
    except ValueError as exc:  # pragma: no cover - solo ante mala configuración.
        raise RuntimeError(f"{name} debe ser un número válido.") from exc

    if parsed < 0:
        return default
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
# Tiempo que se mantendrán las respuestas cacheadas. Un valor bajo reduce la
# presión sobre el LLM cuando se repiten preguntas idénticas.
QA_CACHE_TTL_SECONDS = _read_positive_int("QA_CACHE_TTL_SECONDS", default=300)
# Máximo de turnos que se conservarán por conversación en el backend. No tiene
# por qué coincidir con ``MAX_HISTORY_TURNS`` porque aquí controlamos cuánto
# historial completo queremos persistir.
MAX_STORED_TURNS = _read_positive_int("MAX_STORED_TURNS", default=50)
# Tiempo de vida opcional para cada historial. Si es ``None`` el historial se
# conserva indefinidamente hasta que el usuario lo elimine manualmente.
HISTORY_TTL_SECONDS = _read_positive_int("HISTORY_TTL_SECONDS")
# Límite blando en bytes para mantener los historiales en memoria dentro de
# un presupuesto acotado y evitar que el proceso crezca indefinidamente cuando
# se atienden conversaciones de larga duración.
CHAT_HISTORY_MEMORY_LIMIT_BYTES = _read_positive_int(
    "CHAT_HISTORY_MEMORY_LIMIT_BYTES", default=262_144
)
# Parámetros de resiliencia para el retriever y el modelo generativo.
RETRIEVER_TIMEOUT_SECONDS = _read_positive_int("RETRIEVER_TIMEOUT_SECONDS", default=30)
LLM_TIMEOUT_SECONDS = _read_positive_int("LLM_TIMEOUT_SECONDS", default=60)
RETRIEVER_MAX_RETRIES = _read_positive_int("RETRIEVER_MAX_RETRIES", default=2)
LLM_MAX_RETRIES = _read_positive_int("LLM_MAX_RETRIES", default=1)
RETRY_BACKOFF_SECONDS = _read_non_negative_float("RETRY_BACKOFF_SECONDS", default=0.5)

# Alias de tipos para mantener legibles las anotaciones que utilizan listas de
# pares ``(pregunta, respuesta)``.
ConversationHistory = List[Tuple[str, str]]
T = TypeVar("T")

# Métricas de observabilidad expuestas vía Prometheus/OpenTelemetry.
RETRIEVER_LATENCY_SECONDS = Histogram(
    "retriever_latency_seconds",
    "Duración de las operaciones de recuperación de documentos.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
LLM_LATENCY_SECONDS = Histogram(
    "llm_latency_seconds",
    "Tiempo empleado por el modelo generativo para producir respuestas.",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
RETRIEVER_TIMEOUTS_TOTAL = Counter(
    "retriever_timeouts_total", "Número de timeouts al consultar el retriever."
)
LLM_TIMEOUTS_TOTAL = Counter(
    "llm_timeouts_total", "Número de timeouts al invocar el modelo generativo."
)
RETRIEVER_RETRIES_TOTAL = Counter(
    "retriever_retries_total",
    "Intentos adicionales realizados para recuperar documentos tras un fallo.",
)
LLM_RETRIES_TOTAL = Counter(
    "llm_retries_total",
    "Intentos adicionales realizados tras errores del modelo generativo.",
)
RETRIEVER_FAILURES_TOTAL = Counter(
    "retriever_failures_total",
    "Fallos definitivos del retriever tras agotar los reintentos.",
)
LLM_FAILURES_TOTAL = Counter(
    "llm_failures_total",
    "Fallos definitivos del modelo generativo tras agotar los reintentos.",
)
CHAT_HISTORY_MEMORY_BYTES = Gauge(
    "chat_history_memory_bytes",
    "Uso estimado de memoria (en bytes) del historial pasado al prompt.",
)
CHAT_HISTORY_TRUNCATIONS_TOTAL = Counter(
    "chat_history_truncations_total",
    "Veces que el historial se recortó por superar el límite de memoria.",
)

app = FastAPI(title="PDF Question Answering API")

default_allowed_origins = {
    "http://localhost:51635",
    "http://127.0.0.1:51635",
}
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_raw and allowed_origins_raw.strip() != "*":
    parsed_origins = {
        origin.strip()
        for origin in allowed_origins_raw.split(",")
        if origin.strip()
    }
    allowed_origins = sorted(parsed_origins | default_allowed_origins)
else:
    allowed_origins = sorted(default_allowed_origins)

allowed_methods = ["GET", "POST", "OPTIONS"]
allowed_headers = ["Content-Type", "Accept"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=allowed_methods,
    allow_headers=allowed_headers,
)

# ``qa_chain`` se inicializa en ``None`` y posteriormente se rellenará en el
# evento de inicio de FastAPI. Usar una variable global evita reconstruir toda la
# tubería de LangChain en cada petición HTTP, lo que reduciría el rendimiento.
qa_chain: RetrievalQA | None = None

# ``conversation_store`` proporciona un backend compartido (Redis en este caso)
# donde se almacena de forma segura el historial de preguntas/respuestas por
# usuario y conversación. Esto permite que múltiples instancias del servidor
# compartan contexto sin depender de memoria local.
conversation_store: "ConversationStore" | None = None
# ``response_cache`` almacena respuestas recientes para evitar recalcularlas en
# cada petición cuando el usuario repite exactamente la misma consulta.
response_cache: "ResponseCache" | None = None

# Índice auxiliar que relaciona nombres de archivos PDF con la ruta completa
# guardada en los metadatos. Sirve para realizar filtros sencillos antes de
# consultar el índice vectorial.
DOCUMENT_SOURCE_INDEX: dict[str, str] = {}


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


class ResponseCache(Protocol):
    """Interfaz mínima para los backends de caché de respuestas."""

    def get(self, user_id: str, normalized_question: str) -> Tuple[str, List[str]] | None:
        ...

    def set(
        self,
        user_id: str,
        normalized_question: str,
        answer: str,
        context: List[str],
    ) -> None:
        ...


class ResponseCacheError(RuntimeError):
    """Indica fallos al interactuar con el backend de caché."""


class RedisResponseCache:
    """Caché ligera de respuestas basada en Redis."""

    def __init__(self, client: Redis, *, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _build_key(user_id: str, normalized_question: str) -> str:
        return f"qa-cache:{user_id}:{normalized_question}"

    def get(self, user_id: str, normalized_question: str) -> Tuple[str, List[str]] | None:
        key = self._build_key(user_id, normalized_question)
        try:
            payload = self._client.get(key)
        except RedisError as exc:  # pragma: no cover - solo ante fallos de red.
            raise ResponseCacheError("No fue posible recuperar la respuesta en caché.") from exc

        if not payload:
            return None

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Entrada de caché dañada en Redis para %s", key)
            return None

        answer = data.get("answer")
        context = data.get("context")
        if not isinstance(answer, str) or not isinstance(context, list):
            logger.warning("Formato inesperado en la caché para %s", key)
            return None

        sanitized_context = [str(item) for item in context]
        return answer, sanitized_context

    def set(
        self,
        user_id: str,
        normalized_question: str,
        answer: str,
        context: List[str],
    ) -> None:
        key = self._build_key(user_id, normalized_question)
        payload = json.dumps({"answer": answer, "context": context}, ensure_ascii=False)
        try:
            self._client.setex(key, self._ttl_seconds, payload)
        except RedisError as exc:  # pragma: no cover - solo ante fallos de red.
            raise ResponseCacheError("No fue posible guardar la respuesta en caché.") from exc


def _create_redis_client() -> Redis:
    try:
        client = Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
    except RedisError as exc:  # pragma: no cover - se activa solo si Redis está caído.
        raise ConversationStoreError("No se pudo conectar a Redis para el historial de chats.") from exc
    return client


def create_conversation_store() -> ConversationStore:
    """Inicializa el backend de historiales compartido usando Redis."""

    client = _create_redis_client()
    return RedisConversationStore(
        client,
        max_length=MAX_STORED_TURNS,
        ttl_seconds=HISTORY_TTL_SECONDS,
    )


def create_response_cache() -> ResponseCache | None:
    """Crea la caché de respuestas si la configuración lo permite."""

    if QA_CACHE_TTL_SECONDS is None:
        return None

    try:
        client = _create_redis_client()
    except ConversationStoreError as exc:
        raise ResponseCacheError("No se pudo conectar a Redis para la caché de respuestas.") from exc

    return RedisResponseCache(client, ttl_seconds=QA_CACHE_TTL_SECONDS)

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
            - Si te es posible dirigete al usuario como Halcoamigo.
            - Procura no traducir palabras ingles - español al hacer tu busqueda en el contexto proporcionado.
            - Si la pregunta **no tiene relación con el contexto** o el contexto **no contiene información útil**, responde exactamente:
            <p>No encontré información relacionada en la documentación.</p>
            - Si la respuesta requiere información extensa, **resume solo lo esencial** (máximo 5 líneas de texto total).
            - No cites todo el documento ni fragmentos largos.
            - No uses frases como “según la información proporcionada” ni “de acuerdo al contexto”.
            - Incluye al final de la respuesta los datos de contacto de la persona mencionada en tu respuesta, si no cuentas con la informacion omite esta parte.


            Historial reciente de la conversación (usuario → asistente):
            {chat_history_text}

            "Contexto disponible:\n{context}\n\n"
            "Pregunta del usuario: {question}\n\n"
            "Respuesta en HTML:"""
        ),
        input_variables=["context", "question"],
        partial_variables={"chat_history_text": "(sin historial previo)"},
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


def normalize_question(question: str) -> str:
    """Devuelve una forma canónica de la pregunta para reutilizarla en claves."""

    return " ".join(question.strip().lower().split())


def _discover_document_sources(pdf_dir: str) -> dict[str, str]:
    """Crea un índice rápido de PDF disponibles para filtros por metadatos."""

    index: dict[str, str] = {}
    for path in glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True):
        name = Path(path).stem
        normalized_name = normalize_question(name.replace("_", " ").replace("-", " "))
        if normalized_name:
            index[normalized_name] = path
    return index


def _select_metadata_filter(normalized_question: str) -> dict[str, str] | None:
    """Devuelve un filtro por metadatos si la pregunta cita un PDF concreto."""

    if not DOCUMENT_SOURCE_INDEX:
        return None

    matches = [
        source_path
        for name, source_path in DOCUMENT_SOURCE_INDEX.items()
        if name and name in normalized_question
    ]

    if not matches:
        return None

    for source_path in sorted(set(matches), key=len, reverse=True):
        return {"source": source_path}
    return None


def _calculate_dynamic_k(question: str) -> int:
    """Ajusta el número de documentos recuperados según la longitud de la consulta."""

    word_count = len(question.split())
    if word_count <= 5:
        return 2
    if word_count <= 15:
        return 4
    if word_count <= 40:
        return 6
    return 8


def _build_search_kwargs(question: str) -> dict[str, object]:
    """Calcula parámetros dinámicos para el retriever de LangChain."""

    normalized = normalize_question(question)
    search_kwargs: dict[str, object] = {"k": _calculate_dynamic_k(question)}
    metadata_filter = _select_metadata_filter(normalized)
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return search_kwargs


def _estimate_turn_size(question: str, answer: str) -> int:
    """Calcula el tamaño aproximado de un turno en bytes."""

    return len(question.encode("utf-8")) + len(answer.encode("utf-8"))


def _estimate_history_size(history: ConversationHistory) -> int:
    """Devuelve el tamaño estimado en bytes de todo el historial."""

    return sum(_estimate_turn_size(question, answer) for question, answer in history)


def _trim_history_to_budget(history: ConversationHistory, limit_bytes: int) -> ConversationHistory:
    """Recorta el historial manteniendo los turnos más recientes dentro del límite."""

    if limit_bytes <= 0:
        return history

    trimmed: list[Tuple[str, str]] = []
    total = 0
    for question, answer in reversed(history):
        entry_size = _estimate_turn_size(question, answer)
        trimmed.append((question, answer))
        total += entry_size
        if total >= limit_bytes:
            break
    trimmed.reverse()
    return trimmed or history[-1:]


def _prepare_history_for_prompt(history: ConversationHistory) -> ConversationHistory:
    """Actualiza métricas de memoria y aplica recortes si es necesario."""

    if not history:
        CHAT_HISTORY_MEMORY_BYTES.set(0)
        return history

    estimated_bytes = _estimate_history_size(history)
    CHAT_HISTORY_MEMORY_BYTES.set(float(estimated_bytes))

    limit = CHAT_HISTORY_MEMORY_LIMIT_BYTES
    if limit is not None and estimated_bytes > limit:
        trimmed_history = _trim_history_to_budget(history, limit)
        trimmed_bytes = _estimate_history_size(trimmed_history)
        CHAT_HISTORY_MEMORY_BYTES.set(float(trimmed_bytes))
        CHAT_HISTORY_TRUNCATIONS_TOTAL.inc()
        logger.warning(
            "Historial recortado por límite de memoria: %.0f bytes (límite %s bytes).",
            estimated_bytes,
            limit,
        )
        return trimmed_history

    return history


def _run_with_timeout(operation: Callable[[], T], timeout_seconds: int | None) -> T:
    """Ejecuta ``operation`` con un timeout opcional usando un hilo auxiliar."""

    if timeout_seconds is None:
        return operation()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError("Operation timed out") from exc


def _execute_with_metrics(
    operation: Callable[[], T],
    *,
    latency_metric: Histogram,
    timeout_counter: Counter,
    failure_counter: Counter,
    retry_counter: Counter,
    timeout_seconds: int | None,
    max_retries: int | None,
    operation_name: str,
) -> T:
    """Ejecuta ``operation`` con métricas, reintentos y timeout."""

    attempts = 1 + (max_retries or 0)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            result = _run_with_timeout(operation, timeout_seconds)
            latency_metric.observe(time.perf_counter() - start)
            return result
        except TimeoutError as exc:
            latency_metric.observe(time.perf_counter() - start)
            timeout_counter.inc()
            last_error = exc
        except Exception as exc:  # pragma: no cover - solo ante fallos del backend real.
            latency_metric.observe(time.perf_counter() - start)
            last_error = exc

        if attempt < attempts:
            retry_counter.inc()
            if RETRY_BACKOFF_SECONDS:
                time.sleep(RETRY_BACKOFF_SECONDS)

    failure_counter.inc()
    raise RuntimeError(
        f"{operation_name} falló tras {attempts} intentos."
    ) from last_error


@app.on_event("startup")
def on_startup() -> None:
    """FastAPI startup hook to prepare the LangChain pipeline once when the server launches."""

    # FastAPI ejecuta esta función automáticamente al iniciar el servidor.
    # Construir el ``qa_chain`` aquí garantiza que los recursos pesados (lectura
    # de PDF, generación de embeddings, etc.) se realicen una única vez.
    global qa_chain, conversation_store, response_cache, DOCUMENT_SOURCE_INDEX
    conversation_store = create_conversation_store()
    try:
        response_cache = create_response_cache()
    except ResponseCacheError as exc:
        logger.warning("No se pudo inicializar la caché de respuestas: %s", exc)
        response_cache = None
    DOCUMENT_SOURCE_INDEX = _discover_document_sources(PDF_FOLDER)
    qa_chain = initialize_qa_chain()


def _execute_qa_chain(
    question: str,
    chat_history: ConversationHistory,
    chat_history_text: str,
) -> dict:
    """Run the blocking LangChain pipeline synchronously."""

    if qa_chain is None:  # Defensive: the caller checks beforehand.
        raise RuntimeError("QA chain is not initialized")

    llm_chain = getattr(getattr(qa_chain, "combine_documents_chain", None), "llm_chain", None)

    result = None

    if llm_chain is not None and hasattr(llm_chain, "partial"):
        try:
            llm_with_history = llm_chain.partial(chat_history_text=chat_history_text)
            retriever = getattr(qa_chain, "retriever", None)

            if retriever is not None and hasattr(retriever, "get_relevant_documents"):
                search_kwargs = _build_search_kwargs(question)
                vector_store = getattr(retriever, "vectorstore", None)
                if vector_store is not None and hasattr(vector_store, "similarity_search"):

                    def _search_operation() -> List[Document]:
                        return vector_store.similarity_search(
                            question,
                            k=int(search_kwargs.get("k", 4)),
                            filter=search_kwargs.get("filter"),
                        )

                    source_docs = _execute_with_metrics(
                        _search_operation,
                        latency_metric=RETRIEVER_LATENCY_SECONDS,
                        timeout_counter=RETRIEVER_TIMEOUTS_TOTAL,
                        failure_counter=RETRIEVER_FAILURES_TOTAL,
                        retry_counter=RETRIEVER_RETRIES_TOTAL,
                        timeout_seconds=RETRIEVER_TIMEOUT_SECONDS,
                        max_retries=RETRIEVER_MAX_RETRIES,
                        operation_name="Retriever",
                    )
                else:

                    def _search_operation() -> List[Document]:
                        original_kwargs = getattr(retriever, "search_kwargs", None)
                        merged_kwargs = {}
                        if isinstance(original_kwargs, dict):
                            merged_kwargs.update(original_kwargs)
                        merged_kwargs.update(search_kwargs)
                        if hasattr(retriever, "search_kwargs"):
                            retriever.search_kwargs = merged_kwargs  # type: ignore[attr-defined]
                        try:
                            return retriever.get_relevant_documents(question)
                        finally:
                            if hasattr(retriever, "search_kwargs"):
                                retriever.search_kwargs = original_kwargs  # type: ignore[attr-defined]

                    source_docs = _execute_with_metrics(
                        _search_operation,
                        latency_metric=RETRIEVER_LATENCY_SECONDS,
                        timeout_counter=RETRIEVER_TIMEOUTS_TOTAL,
                        failure_counter=RETRIEVER_FAILURES_TOTAL,
                        retry_counter=RETRIEVER_RETRIES_TOTAL,
                        timeout_seconds=RETRIEVER_TIMEOUT_SECONDS,
                        max_retries=RETRIEVER_MAX_RETRIES,
                        operation_name="Retriever",
                    )
                context = "\n\n".join((doc.page_content or "") for doc in source_docs)
                
                def _llm_operation() -> str:
                    return llm_with_history.run(
                        context=context,
                        question=question,
                    )

                answer_text = _execute_with_metrics(
                    _llm_operation,
                    latency_metric=LLM_LATENCY_SECONDS,
                    timeout_counter=LLM_TIMEOUTS_TOTAL,
                    failure_counter=LLM_FAILURES_TOTAL,
                    retry_counter=LLM_RETRIES_TOTAL,
                    timeout_seconds=LLM_TIMEOUT_SECONDS,
                    max_retries=LLM_MAX_RETRIES,
                    operation_name="LLM",
                )
                result = {
                    "result": answer_text,
                    "source_documents": source_docs,
                }
        except Exception as exc:  # pragma: no cover - solo ante fallos en la cadena real.
            logger.exception("Fallo al ejecutar la cadena con historial parcial: %s", exc)

    if result is None:
        def _fallback_operation() -> dict:
            return qa_chain(
                {
                    "query": question,
                    "chat_history": chat_history,
                    "chat_history_text": chat_history_text,
                }
            )

        result = _execute_with_metrics(
            _fallback_operation,
            latency_metric=LLM_LATENCY_SECONDS,
            timeout_counter=LLM_TIMEOUTS_TOTAL,
            failure_counter=LLM_FAILURES_TOTAL,
            retry_counter=LLM_RETRIES_TOTAL,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
            operation_name="LLM fallback",
        )
    return result


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
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
    normalized_question = normalize_question(payload.question)
    if conversation_store is None:
        raise HTTPException(status_code=503, detail="Conversation store is not ready yet.")

    cached_entry: Tuple[str, List[str]] | None = None
    if response_cache is not None:
        try:
            cached_entry = await run_in_threadpool(
                response_cache.get,
                payload.user_id,
                normalized_question,
            )
        except ResponseCacheError as exc:
            logger.warning("No se pudo consultar la caché de respuestas: %s", exc)

    if cached_entry is not None:
        answer, cached_context = cached_entry
        try:
            await run_in_threadpool(
                conversation_store.append_turn,
                payload.user_id,
                conversation_id,
                payload.question,
                answer,
            )
        except ConversationStoreError as exc:
            logger.error("Fallo al guardar el historial (respuesta en caché): %s", exc)
            raise HTTPException(status_code=503, detail="Failed to persist conversation history.") from exc

        return AskResponse(
            answer=answer,
            context=cached_context,
            conversation_id=conversation_id,
        )

    try:
        # Se recupera el historial desde Redis para que todos los workers
        # compartan exactamente el mismo contexto.
        chat_history = await run_in_threadpool(
            conversation_store.load_history, payload.user_id, conversation_id
        )
    except ConversationStoreError as exc:
        logger.error("Fallo al recuperar el historial: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to load conversation history.") from exc

    chat_history = _prepare_history_for_prompt(chat_history)

    # Run the retrieval QA chain to obtain both the answer and the source documents.
    # El objeto ``qa_chain`` se comporta como una función: recibe un diccionario
    # con la clave "query" y devuelve otro diccionario que incluye la respuesta
    # generada y los documentos relevantes.
    chat_history_text = summarize_chat_history(chat_history)

    # ``RetrievalQA`` no reconoce variables adicionales en cada llamada. Para
    # poder incluir el historial resumido en el prompt reutilizamos la cadena
    # de forma segura creando un ``partial`` independiente por petición en lugar
    # de modificar el prompt global (lo que provocaba condiciones de carrera
    # bajo carga). Si la optimización falla por cualquier motivo, se recurre al
    # flujo estándar de ``qa_chain`` como mecanismo de respaldo.
    result = await run_in_threadpool(
        _execute_qa_chain,
        payload.question,
        chat_history,
        chat_history_text,
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

    # Actualiza el historial almacenando la nueva pareja pregunta/respuesta en
    # el backend compartido. ``append_turn`` usa operaciones atómicas (RPUSH +
    # LTRIM) para evitar condiciones de carrera entre múltiples peticiones.
    try:
        await run_in_threadpool(
            conversation_store.append_turn,
            payload.user_id,
            conversation_id,
            payload.question,
            answer,
        )
    except ConversationStoreError as exc:
        logger.error("Fallo al guardar el historial: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to persist conversation history.") from exc

    if response_cache is not None:
        try:
            await run_in_threadpool(
                response_cache.set,
                payload.user_id,
                normalized_question,
                answer,
                context_snippets,
            )
        except ResponseCacheError as exc:
            logger.warning("No se pudo guardar la respuesta en caché: %s", exc)

    return AskResponse(
        answer=answer,
        context=context_snippets,
        conversation_id=conversation_id,
    )


@app.get("/metrics")
def metrics() -> Response:
    """Exponer las métricas en formato Prometheus/OpenTelemetry."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root() -> dict[str, str]:
    """Simple health check endpoint."""

    # Responder con un objeto simple permite a servicios externos verificar que
    # la API está viva sin ejecutar el proceso completo de pregunta/respuesta.
    return {"status": "ok"}


@app.get("/tu_ruta")
def example_route() -> dict[str, str]:
    """Endpoint adicional para responder a peticiones GET de prueba."""

    return {"message": "Solicitud recibida correctamente"}


if __name__ == "__main__":
    import uvicorn

    # Permite ejecutar la aplicación directamente con ``python main.py`` durante
    # el desarrollo. Uvicorn es el servidor ASGI recomendado para FastAPI.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
