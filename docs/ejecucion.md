# Guía de Ejecución — mcp-videos

> **Proyecto:** `mcp-videos` · **Fecha:** 2026-10-28 · **Autor:** Alex Prieto Romani <alexprieto1997@gmail.com>

---

## Identifica tu tipo de proyecto

| Tipo | Señal de que aplica | Secciones obligatorias |
|---|---|---|
| [X] **Pipeline / ETL secuencial + Servidor MCP & Preview Web** | Flujos encadenados (Audio → Whisper → HTML → Render 60 FPS) + `FastMCP` + `FastAPI` | §1, §2B, §3, §4C, §4F, §5C, §6 |

---

## 1. Requisitos previos

### 1.1 Software

| Software | Versión mínima | Verificar con |
|---|---|---|
| Python | 3.11+ | `python3 --version` |
| `uv` | 0.5+ | `uv --version` |
| FFmpeg & FFprobe | 5.0+ | `ffmpeg -version` / `ffprobe -version` |
| Playwright Chromium | 1.48+ | `uv run playwright --version` |
| Git | 2.30+ | `git --version` |

### 1.2 Credenciales y accesos

- **Ejecución 100% Local (Por defecto):** No requiere llaves de API externas. Utiliza `kokoro-onnx` / `piper` / sintetizador local para voces predefinidas, `XTTS-v2` / `F5-TTS` para clonación zero-shot desde muestras `.wav`, `faster-whisper` (`int8` en CPU) para alineación palabra por palabra y `Ollama` (`http://localhost:11434/v1`) o plantillas paramétricas locales para las escenas HTML.
- **Proveedores Cloud Opcionales (`.env`):**
  - `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY`: para generación de escenas HTML con modelos en la nube.
  - `ELEVENLABS_API_KEY`: para síntesis de voz comercial por API.

---

## 2. Entorno virtual

### 2B. uv

```bash
# Primera vez — crear entorno e instalar dependencias
uv venv
uv sync

# Instalar el navegador Chromium Headless para validación DOM/JS y renderizado 60 FPS
uv run playwright install chromium

# Verificar estado de herramientas del sistema (ffmpeg, ffprobe, directorios)
uv run python -m mcp_videos.cli doctor

# Ejecutar cualquier comando sin activar manualmente el entorno (recomendado)
uv run python -m mcp_videos.cli --help
```

> **Windows / AppLocker:** si los binarios del entorno están bloqueados en estaciones corporativas, usa siempre el patrón módulo `uv run python -m mcp_videos.<modulo>` en vez del binario directo.
>
> **OneDrive:** si el repositorio llegara a alojarse en una carpeta sincronizada, define `$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\mcp-videos-venv"` antes de ejecutar `uv venv`.

---

## 3. Variables de entorno

```bash
# Copiar plantilla y ajustar si se desean conectar LLMs o TTS por API
cp .env.example .env
```

| Variable | Descripción | Dónde obtenerla |
|---|---|---|
| `MCP_VIDEOS_DB_PATH` | Ruta al archivo SQLite transaccional (`data/mcp_videos.sqlite`) | Generado automáticamente al iniciar |
| `MCP_VIDEOS_WORKSPACES_DIR` | Carpeta raíz donde se guardan audios, escenas `.html` y `.mp4` (`workspaces`) | Local en el repositorio |
| `MCP_VIDEOS_VOICES_DIR` | Carpeta donde se normalizan las muestras `.wav` para clonación (`data/voices`) | Local en el repositorio |
| `LOCAL_INFERENCE_DEVICE` | Dispositivo para TTS y Whisper (`auto`, `cpu`, `cuda`) | Configuración local (`auto` detecta hardware) |
| `WHISPER_MODEL_SIZE` | Tamaño del modelo `faster-whisper` (`tiny`, `base`, `small`, `medium`) | Descarga automática de HuggingFace Hub |
| `WHISPER_COMPUTE_TYPE` | Precisión de inferencia (`int8` para CPU/Iris Xe, `float16` para GPU CUDA) | Configuración local |
| `LOCAL_LLM_BASE_URL` | Endpoint compatible con OpenAI para LLMs locales (`http://localhost:11434/v1`) | Servidor local Ollama / LM Studio / vLLM |
| `LOCAL_LLM_MODEL` | Modelo local para generar código HTML/SVG (`qwen2.5-coder:14b`) | Catálogo local de `ollama list` |
| `ANTHROPIC_API_KEY` | Token opcional para usar Claude en la generación de motion graphics HTML | Console de Anthropic (`.env` local) |
| `GEMINI_API_KEY` | Token opcional para usar Gemini en guiones y escenas HTML | Google AI Studio (`.env` local) |
| `ELEVENLABS_API_KEY` | Token opcional para síntesis de voz vía API ElevenLabs | Portal de ElevenLabs (`.env` local) |
| `PREVIEW_PORT` | Puerto HTTP del Estudio de Previsualización Web (`8080`) | Configuración local |
| `DEFAULT_VIDEO_FPS` | Cuadros por segundo deterministas (`60`, `30` o `24`) | Configuración local |
| `FFMPEG_CRF` | Factor de calidad constante H.264 (`18` = calidad visual sin pérdida aparente) | Configuración local |

> Las credenciales nunca se versionan. El archivo `.env` está incluido en `.gitignore`.

---

## 4. Ejecución

### 4C. Servidor MCP y Estudio Web de Previsualización

**Opción 1 — Levantar el Servidor MCP (para Claude Code, Antigravity, OpenCode o Cursor):**
```bash
uv run python -m mcp_videos.mcp_server
```

**Opción 2 — Levantar el Estudio Web de Previsualización Interactiva (`Live Studio Preview`):**
```bash
uv run python -m uvicorn mcp_videos.preview.server:app --host 127.0.0.1 --port 8080 --reload \
  --reload-exclude ".venv" --reload-exclude "workspaces" --reload-exclude "docs"
```

**Puertos y accesos:**
- Estudio Web: `http://localhost:8080`
- Documentación OpenAPI del servidor de preview: `http://localhost:8080/docs`

**Verificación rápida:**
- [ ] `http://localhost:8080/api/projects` responde `200 OK` con la lista de proyectos en SQLite.
- [ ] El visor carga la escena `.html` en el `iframe` `1080x1920` y sincroniza el audio al mover la barra de tiempo o hacer clic sobre una palabra.
- [ ] No aparecen errores en la consola del navegador.

---

### 4F. Pipeline secuencial por CLI (Creación y Renderizado de un Short)

```bash
# Opción A: Construir y renderizar el Short demostrativo completo en un solo comando
uv run python -m mcp_videos.cli build-demo --project-id short-antena-st5
```

```bash
# Opción B: Ejecución paso a paso por flujo
# [Paso 1/5] — Registrar voces (predefinidas o clonadas desde archivo .wav de 3-30s)
uv run python -m mcp_videos.cli register-voice \
  --voice-id clon-host --name "Host Clonado" --engine f5 --mode clone --sample path/to/sample.wav

# [Paso 2/5] — Crear el proyecto vertical (1080x1920 @ 60 FPS)
uv run python -m mcp_videos.cli create-project \
  --project-id antena-nasa --title "La antena más rara que fue al espacio" --width 1080 --height 1920 --fps 60

# [Paso 3/5] — Sintetizar el diálogo multivoz (-16 LUFS) y extraer word_timestamps con faster-whisper
uv run python -m mcp_videos.cli synthesize \
  --project-id antena-nasa --script-file script_dialogue.json

# [Paso 4/5] — Generar o registrar escenas HTML Motion Graphics y validarlas en Chromium Headless
uv run python -m mcp_videos.cli save-scene \
  --project-id antena-nasa --scene-index 0 --start-ms 0 --end-ms 4500 --html-file scene_00.html

# [Paso 5/5] — Renderizar el video determinista a 60 FPS + subtítulos dinámicos
uv run python -m mcp_videos.cli render --project-id antena-nasa --burn-subtitles
```

**Dependencias entre pasos del pipeline:**

| Paso | Comando / Módulo | Entrada | Salida | Depende de |
|---|---|---|---|---|
| 1 | `audio.voice_registry` (`Flujo 1`) | Muestra `.wav` (3–30 s) o preset ID | Fila en `voices` + `data/voices/<id>.wav` | — |
| 2 | `core.orchestrator.create_project` | Metadatos (`id`, `width`, `height`, `fps`) | Fila en `projects` + carpeta `workspaces/<id>/` | — |
| 3 | `audio.dialogue_synth` + `audio.aligner` (`Flujos 2 y 3`) | Guion por turnos + `voices` | `dialogue_master.wav` + `dialogue_lines` + `word_timestamps` | Pasos 1 y 2 |
| 4 | `scenes.generator` + `scenes.validator` (`Flujo 4`) | `word_timestamps` + código/prompt HTML | `scenes/scene_XX.html` con `validation_status='valid'` | Paso 3 |
| 5 | `render.engine` (`Flujo 6`) | Escenas validadas + `dialogue_master.wav` | `workspaces/<id>/output/final.mp4` (`60 FPS`) | Paso 4 |

---

## 5. Despliegue

### 5C. Ejecución en Servidor Local / Contenedor Docker

```bash
# Construir imagen con Python 3.11, FFmpeg y Chromium Headless
docker build -t mcp-videos:latest .

# Ejecutar Estudio Web y API montando los volúmenes de datos y workspaces
docker run -d -p 8080:8080 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/workspaces:/app/workspaces" \
  --env-file .env \
  mcp-videos:latest
```

---

## 6. Troubleshooting

| Problema | Causa probable | Solución |
|---|---|---|
| `Executable doesn't exist` al validar escena o renderizar | Falta instalar el binario de Chromium para Playwright | Ejecutar `uv run playwright install chromium` |
| `ffmpeg: command not found` en síntesis o renderizado | `ffmpeg` no está instalado en el sistema operativo | Instalar FFmpeg (`sudo apt install ffmpeg` o `pacman -S ffmpeg`) y verificar con `uv run python -m mcp_videos.cli doctor` |
| `SceneNotValidatedError` al intentar exportar a MP4 | Una escena falló la validación DOM/JS o el diálogo se regeneró después de crear la escena | Revisar `validation_errors_json` en SQLite o reejecutar `save-scene` para validar la escena frente a los nuevos timestamps |
| Síntesis lenta en máquinas sin GPU NVIDIA | Se seleccionó un motor pesado en `float32` sobre CPU | Definir `LOCAL_INFERENCE_DEVICE=cpu` y `WHISPER_COMPUTE_TYPE=int8` en `.env`, y usar voces `kokoro`/`piper` para iteración rápida |
| Puerto `8080` ocupado al abrir el Estudio Web | Otro proceso escucha en el puerto por defecto | Cambiar `PREVIEW_PORT=8081` en `.env` o pasar `--port 8081` al arrancar |
