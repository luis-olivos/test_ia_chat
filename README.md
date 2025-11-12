# API de preguntas sobre PDFs

Este proyecto expone una API basada en FastAPI que utiliza LangChain y Gemini para responder preguntas en español a partir de colecciones de documentos PDF. La aplicación carga los PDFs, construye un índice semántico con Chroma y conserva el historial de conversación en Redis para mejorar las respuestas en sesiones posteriores.

## Requisitos previos

- Python 3.10 o superior.
- Dependencias del proyecto instaladas mediante `pip install -r requirements.txt`.
- Credenciales válidas de Google Gemini (variable `GOOGLE_API_KEY`).
- Servidor Redis accesible para almacenar historiales y la caché de respuestas.

## Instalación rápida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A continuación crea un archivo `.env` en la raíz del proyecto (o exporta las variables en tu shell) con la configuración necesaria antes de iniciar la aplicación.

## Preparar los documentos y el índice

1. Coloca tus archivos PDF en la carpeta indicada por `PDF_FOLDER` (por defecto `pdfs/`).
2. Opcionalmente, ejecuta el generador de índice para precalcular los embeddings y acelerar el arranque del servidor:

   ```bash
   python offline_index.py --pdf-folder pdfs --chroma-dir chroma_store
   ```

   Usa `--keep-existing` si deseas conservar un índice previo y solo añadir nuevos documentos.

Si omites este paso, la API construirá el índice automáticamente durante el arranque, lo cual puede tardar más según la cantidad de documentos.

## Variables de entorno principales

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `GOOGLE_API_KEY` | Clave obligatoria para acceder a Gemini. | _sin valor_ |
| `PDF_FOLDER` | Carpeta donde se buscan los PDFs a indexar. | `pdfs` |
| `CHROMA_DIR` | Directorio donde se persiste el índice vectorial. | `chroma_store` |
| `REDIS_URL` | URL de conexión a Redis para historiales y caché. | `redis://localhost:6379/0` |
| `QA_CACHE_TTL_SECONDS` | Tiempo en segundos que se mantiene una respuesta en la caché. Si se define como 0 o negativo, se desactiva la caché. | `300` |
| `MAX_STORED_TURNS` | Número máximo de pares pregunta/respuesta guardados por conversación en Redis. | `50` |
| `HISTORY_TTL_SECONDS` | Tiempo de vida opcional del historial en Redis. Si se omite, el historial permanece indefinidamente. | _sin valor_ |
| `CHAT_HISTORY_MEMORY_LIMIT_BYTES` | Límite aproximado del tamaño del historial que se envía al modelo. | `262144` |
| `RETRIEVER_TIMEOUT_SECONDS` | Tiempo máximo en segundos para recuperar documentos del índice. | `30` |
| `LLM_TIMEOUT_SECONDS` | Tiempo máximo en segundos para obtener la respuesta de Gemini. | `60` |

Consulta `main.py` para más variables avanzadas relacionadas con reintentos y backoff.

## Ejecutar la API

Con el entorno virtual activo y las variables configuradas, inicia el servidor con Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

La opción `--reload` es útil en desarrollo porque recarga la aplicación al detectar cambios en el código. En producción se recomienda ejecutar Uvicorn o Gunicorn sin recarga automática y detrás de un proxy inverso.

## Endpoints disponibles

### `GET /`

Devuelve un objeto JSON simple para verificar que el servicio está en línea.

**Respuesta de ejemplo**

```json
{"status": "ok"}
```

### `POST /ask`

Genera una respuesta basada en los PDFs indexados. Requiere un identificador de usuario para mantener historiales separados y permite un identificador de conversación opcional.

**Cuerpo de la petición**

```json
{
  "question": "¿Cuál es el objetivo principal del documento?",
  "user_id": "usuario-123",
  "conversation_id": "reunion-abril"
}
```

**Respuesta de ejemplo**

```json
{
  "answer": "El objetivo principal es describir la arquitectura propuesta...",
  "context": [
    "Source: pdfs/documento.pdf (page 2)\nEl informe detalla...",
    "Source: pdfs/documento.pdf (page 5)\nLa arquitectura considera..."
  ],
  "conversation_id": "reunion-abril"
}
```

- `answer`: texto generado por Gemini.
- `context`: fragmentos relevantes de los PDFs utilizados como soporte. Cada elemento incluye la ruta y la página.
- `conversation_id`: identificador efectivo utilizado para persistir el historial. Si no se envía en la solicitud, se devuelve `default`.

Si existe una respuesta en caché para la misma pregunta normalizada y usuario, el servicio devolverá la versión almacenada sin volver a consultar al modelo.

### `GET /metrics`

Expone métricas en formato Prometheus/OpenTelemetry, incluyendo latencias del recuperador y del modelo, número de reintentos y uso de memoria del historial. Integra estas métricas con tu stack de observabilidad preferido para monitorear el rendimiento del servicio.

## Historiales y caché

- **Historiales**: cada vez que se recibe una pregunta, la API recupera las interacciones previas desde Redis, resume el contexto y lo envía junto con la nueva consulta. Esto permite respuestas coherentes en conversaciones prolongadas.
- **Caché de respuestas**: si `QA_CACHE_TTL_SECONDS` está configurada, las respuestas se almacenan temporalmente y se reutilizan para preguntas repetidas del mismo usuario. Esto reduce el costo de invocación al modelo y mejora la latencia.

## Buenas prácticas para despliegue

- Utiliza un volumen persistente para `CHROMA_DIR` cuando ejecutes la API en contenedores, garantizando que el índice sobreviva a reinicios.
- Configura Redis en alta disponibilidad o con persistencia según tus necesidades, ya que la API depende de él para los historiales.
- Protege el endpoint `/metrics` detrás de autenticación o una red interna si tus métricas contienen información sensible.
- Habilita TLS y autenticación en la capa de infraestructura (por ejemplo, mediante un proxy inverso) para proteger las credenciales y las preguntas de los usuarios.

Con estos pasos estarás listo para integrar la API en tus aplicaciones y ofrecer respuestas basadas en tus documentos PDF internos.
