# Guía de Ejecución — mcp-videos

> **Proyecto:** `mcp-videos` · **Fecha:** 2026-10-28 (Actualizado Fase 6) · **Autor:** Alex Prieto Romani <alexprieto1997@gmail.com>

---

## Identifica tu tipo de proyecto

| Tipo | Señal de que aplica | Secciones obligatorias |
|---|---|---|
| [X] **Pipeline / ETL secuencial + Servidor MCP & Preview Web** | Flujos encadenados (Audio-First → Whisper → HTML → Auditoría Visual → Render Híbrido 60 FPS) + `FastMCP` + `FastAPI` | §1, §2B, §3, §4C, §4F, §5C, §6 |

---

## 1. Requisitos previos

### 1.1 Software

| Software | Versión mínima | Verificar con | Notas |
|---|---|---|---|
| Python | 3.11+ | `python3 --version` | Núcleo de dominio, servidor MCP y CLI |
| `uv` | 0.5+ | `uv --version` | Gestor de paquetes y entornos virtuales ultrarrápido |
| FFmpeg & FFprobe | 5.0+ | `ffmpeg -version` / `ffprobe -version` | Normalización LUFS, concat de audio y contenedor MP4 |
| Playwright Chromium | 1.48+ | `uv run playwright --version` | Motor base para validación DOM/JS y render determinista Playwright |
| Git | 2.30+ | `git --version` | Control de versiones del proyecto |
| Node.js & npm *(Opcional)* | 18.0+ | `node --version` / `npm --version` | Requerido para el motor acelerado HyperFrames |
| HyperFrames CLI *(Opcional)* | 0.1+ | `hyperframes --version` | Motor multi-worker GPU (~4.5x render) y 80+ checks de auditoría visual |

#### 1.1.1 Habilitación Opcional del Motor Multi-Worker HyperFrames

`mcp-videos` implementa una arquitectura híbrida (**ADR-06**). Aunque el sistema opera de forma 100% autónoma utilizando únicamente Python, FFmpeg y **Playwright**, es posible habilitar opcionalmente el motor **HyperFrames** para acelerar el renderizado hasta ~4.5x (mediante un pool concurrente de workers Chromium con GPU y protocolo CDP `BeginFrame`) y habilitar una suite de más de 80 comprobaciones visuales automatizadas (colisión de elementos, desbordamiento de viewport, contrastes WCAG y sincronización):

```bash
# Opción A: Instalación global con npm (recomendada si se dispone de Node.js)
npm install -g hyperframes

# Opción B: Instalación en el directorio local del proyecto
npm install --save-dev hyperframes

# Opción C: Instalación en una ruta personalizada externa
# Configure la variable MCP_VIDEOS_HYPERFRAMES_DIR en su .env:
# MCP_VIDEOS_HYPERFRAMES_DIR="C:\herramientas\node_modules"   # En Windows
# MCP_VIDEOS_HYPERFRAMES_DIR="/opt/hyperframes/node_modules" # En Linux/macOS
```

> **Degradación elegante:** Si HyperFrames no está presente en el sistema, `mcp-videos` aplica fallback automático y transparente hacia el motor monohilo Playwright + FFmpeg sin interrumpir ningún flujo.

### 1.2 Credenciales y accesos

- **Ejecución 100% Local (Por defecto):** No requiere llaves de API externas ni conexión a internet para operar. Utiliza `kokoro-onnx` / `piper` para voces predefinidas en CPU, `XTTS-v2` / `F5-TTS` para clonación zero-shot desde muestras `.wav`, `faster-whisper` (`int8` en CPU o GPU) para alineación temporal palabra a palabra y `Ollama` (`http://localhost:11434/v1`) o plantillas paramétricas locales para las escenas HTML.
- **Proveedores Cloud Opcionales (`.env`):**
  - `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY`: para generación y ajuste de escenas HTML con modelos en la nube.
  - `ELEVENLABS_API_KEY`: para síntesis de voz comercial por API.

---

## 2. Entorno virtual

### 2B. uv

```bash
# Primera vez — crear entorno e instalar dependencias
uv venv
uv sync

# Instalar el navegador Chromium Headless para validación DOM/JS y fallback de render
uv run playwright install chromium

# Diagnosticar dependencias externas del sistema (ffmpeg, ffprobe, chromium, hyperframes)
uv run python -m mcp_videos.cli doctor

# Diagnosticar en formato JSON estructurado (útil para agentes IA o integración continua)
uv run python -m mcp_videos.cli doctor --json

# Ejecutar ayuda general del CLI
uv run python -m mcp_videos.cli --help
```

#### Detección Inteligente Multiplataforma (`doctor`)

El comando `uv run python -m mcp_videos.cli doctor` (o `mcp-videos doctor --json`) analiza automáticamente el entorno y localiza:
- **`ffmpeg` y `ffprobe`:** localizados en `PATH` o rutas estándar del sistema operativo.
- **`chromium`:** inspecciona variables de entorno (`MCP_VIDEOS_CHROMIUM_PATH`, `PLAYWRIGHT_BROWSERS_PATH`), ejecutables en `PATH`, Google Chrome del sistema en Windows (`C:\Program Files\Google\Chrome\Application\chrome.exe`) y macOS, así como las rutas estándar de caché de Playwright:
  - **Windows:** `%LOCALAPPDATA%\ms-playwright`
  - **Linux:** `~/.cache/ms-playwright`
  - **macOS:** `~/Library/Caches/ms-playwright`
- **`hyperframes`:** detecta el binario global en `PATH` (`hyperframes`), el directorio configurado en `MCP_VIDEOS_HYPERFRAMES_DIR`, `node_modules/.bin` local o `npx`. Si no está instalado, se clasifica como `NOT FOUND (opcional)` sin degradar el estado de salud (`healthy: true`), manteniendo el sistema listo para operar con Playwright.

**Ejemplo de salida de `doctor`:**
```text
mcp-videos 0.1.0 — doctor
Python: 3.11.9
ffmpeg: OK (C:\ffmpeg\bin\ffmpeg.exe)
ffprobe: OK (C:\ffmpeg\bin\ffprobe.exe)
chromium: OK (C:\Users\...\AppData\Local\ms-playwright\chromium-1148\chrome-win\chrome.exe)
hyperframes: OK (C:\Users\...\AppData\Roaming\npm\hyperframes.cmd)
device: cpu
preview_port: 8080
status: OK
```

> **Windows / AppLocker:** Si los binarios `.exe` dentro del entorno virtual están bloqueados por políticas de seguridad corporativas, utiliza siempre la invocación por módulo: `uv run python -m mcp_videos.cli <comando>`.
>
> **OneDrive:** Si el repositorio se aloja en una carpeta sincronizada, define `$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\mcp-videos-venv"` antes de ejecutar `uv venv`.

---

## 3. Variables de entorno

```bash
# Copiar plantilla y ajustar parámetros si se desean conectar LLMs o TTS por API
cp .env.example .env
```

| Variable | Descripción | Dónde obtenerla |
|---|---|---|
| `MCP_VIDEOS_DB_PATH` | Ruta al archivo SQLite transaccional (`data/mcp_videos.sqlite`) | Generado automáticamente al iniciar |
| `MCP_VIDEOS_WORKSPACES_DIR` | Carpeta raíz donde se guardan audios, escenas `.html`, composiciones y `.mp4` (`workspaces`) | Local en el repositorio |
| `MCP_VIDEOS_VOICES_DIR` | Carpeta donde se normalizan las muestras `.wav` para clonación (`data/voices`) | Local en el repositorio |
| `LOCAL_INFERENCE_DEVICE` | Dispositivo para TTS y Whisper (`auto`, `cpu`, `cuda`) | Configuración local (`auto` detecta hardware disponible) |
| `WHISPER_MODEL_SIZE` | Tamaño del modelo `faster-whisper` (`tiny`, `base`, `small`, `medium`) | Descarga automática desde HuggingFace Hub |
| `WHISPER_COMPUTE_TYPE` | Precisión de inferencia (`int8` para CPU/Iris Xe, `float16` para GPU CUDA) | Configuración local |
| `LOCAL_LLM_BASE_URL` | Endpoint compatible con OpenAI para LLMs locales (`http://localhost:11434/v1`) | Servidor local Ollama / LM Studio / vLLM |
| `LOCAL_LLM_MODEL` | Modelo local para generar código HTML/SVG (`qwen2.5-coder:14b`) | Catálogo local de `ollama list` |
| `DEFAULT_RENDER_BACKEND` | Motor de renderizado por defecto (`auto`, `hyperframes`, `playwright`) | Configuración local (`auto` por defecto) |
| `MCP_VIDEOS_HYPERFRAMES_DIR` | Carpeta base donde reside HyperFrames (`node_modules` o raíz de instalación) | Local (opcional si se instaló localmente) |
| `MCP_VIDEOS_CHROMIUM_PATH` | Ruta personalizada hacia un binario de Chromium o Chrome específico | Configuración local opcional |
| `ANTHROPIC_API_KEY` | Token opcional para usar Claude en la generación de motion graphics HTML | Console de Anthropic (`.env` local) |
| `GEMINI_API_KEY` | Token opcional para usar Gemini en guiones y escenas HTML | Google AI Studio (`.env` local) |
| `ELEVENLABS_API_KEY` | Token opcional para síntesis de voz vía API ElevenLabs | Portal de ElevenLabs (`.env` local) |
| `PREVIEW_PORT` | Puerto HTTP del Estudio de Previsualización Web (`8080`) | Configuración local |
| `DEFAULT_VIDEO_FPS` | Cuadros por segundo deterministas (`60`, `30` o `24`) | Configuración local (60 FPS recomendado) |
| `FFMPEG_CRF` | Factor de calidad constante H.264 (`18` = alta fidelidad sin pérdida perceptible) | Configuración local |

> Las credenciales nunca se versionan en el repositorio. El archivo `.env` está estrictamente ignorado por `.gitignore`.

---

## 4. Ejecución

### 4C. Servidor MCP y Estudio Web de Previsualización

#### Opción 1 — Levantar el Servidor MCP (para Claude Code, Antigravity, OpenCode o Cursor)

```bash
uv run python -m mcp_videos.mcp_server
```

**Herramientas MCP expuestas para agentes IA (Fase 6):**
- `register_voice(voice_id, name, engine, mode, sample_wav_path, language, config)`: Registra una voz predefinida o clona a partir de un WAV de 3–30 s.
- `list_voices(language)`: Lista el catálogo de voces disponibles.
- `create_project(project_id, title, width, height, fps, description)`: Inicializa el proyecto y su workspace.
- `synthesize_dialogue(project_id, lines)`: Sintetiza el diálogo multivoz (-16 LUFS), genera `dialogue_master.wav` y puebla `word_timestamps` con `faster-whisper`.
- `save_and_validate_scene(project_id, scene_index, title, start_ms, end_ms, html_content, prompt, provider)`: Guarda y valida una escena HTML con compuerta DOM/JS en Playwright.
- `audit_project_visuals(project_id, backend="auto")`: **(Nuevo en Fase 6)** Exporta la composición híbrida (`index.html`) con el arnés dual `window.__timelines` ↔ `window.__setFrameTime` y ejecuta más de 80 comprobaciones visuales con HyperFrames (o validación DOM/JS en Playwright si `backend="playwright"` o no está instalado).
- `render_project_video(project_id, burn_subtitles=True, fps=None, backend="auto")`: **(Actualizado en Fase 6)** Renderiza el video determinista a FPS constantes en `final.mp4`. Soporta `backend="auto"|"hyperframes"|"playwright"`.

#### Opción 2 — Levantar el Estudio Web de Previsualización Interactiva (`Live Studio Preview`)

```bash
uv run python -m uvicorn mcp_videos.preview.server:app --host 127.0.0.1 --port 8080 --reload \
  --reload-exclude ".venv" --reload-exclude "workspaces" --reload-exclude "docs"
```

**Puertos y accesos:**
- Estudio Web: `http://localhost:8080`
- Documentación interactiva OpenAPI: `http://localhost:8080/docs`

**Verificación rápida:**
- [ ] `http://localhost:8080/api/projects` responde `200 OK` con la lista de proyectos en SQLite.
- [ ] El reproductor monta la escena en el `iframe` `1080x1920`, avanza el reloj virtual `__setFrameTime` y resalta la palabra activa sin errores en la consola web.

---

### 4F. Pipeline secuencial por CLI (Creación, Auditoría y Renderizado Híbrido)

#### Opción A: Construir y renderizar el Short demostrativo completo en un solo comando

```bash
uv run python -m mcp_videos.cli build-demo --project-id short-antena-st6
```

#### Opción B: Ejecución paso a paso por flujo (CLI Fase 6)

```bash
# [Paso 1/6] — Registrar voces (predefinidas o clonadas desde archivo .wav de 3-30s)
uv run python -m mcp_videos.cli register-voice \
  --voice-id clon-host --name "Host Clonado" --engine f5 --mode clone --sample path/to/sample.wav

# [Paso 2/6] — Crear el proyecto vertical (1080x1920 @ 60 FPS)
uv run python -m mcp_videos.cli create-project \
  --project-id antena-nasa --title "La antena más rara que fue al espacio" --width 1080 --height 1920 --fps 60

# [Paso 3/6] — Sintetizar el diálogo multivoz (-16 LUFS) y extraer word_timestamps con faster-whisper
uv run python -m mcp_videos.cli synthesize \
  --project-id antena-nasa --script-file script_dialogue.json

# [Paso 4/6] — Generar o registrar escenas HTML Motion Graphics y validarlas en Chromium Headless
uv run python -m mcp_videos.cli save-scene \
  --project-id antena-nasa --scene-index 0 --start-ms 0 --end-ms 4500 --html-file scene_00.html

# [Paso 5/6] — Auditar visualmente el proyecto con HyperFrames (80+ checks de layout y contraste)
uv run python -m mcp_videos.cli check \
  --project-id antena-nasa --backend auto

# Opcional: salida estructurada en JSON para scripts de automatización
uv run python -m mcp_videos.cli check \
  --project-id antena-nasa --backend auto --json

# [Paso 6/6] — Renderizar el video determinista a 60 FPS con motor híbrido
uv run python -m mcp_videos.cli render \
  --project-id antena-nasa --backend auto --burn-subtitles
```

#### Modos de Backend para Renderizado y Auditoría (`--backend`)

- `auto` *(por defecto)*: Detecta automáticamente si HyperFrames está disponible en el entorno. Si lo encuentra, delega en el motor multi-worker acelerado por GPU; si no, realiza un fallback transparente y limpio a Playwright monohilo.
- `hyperframes`: Fuerza el uso de `@hyperframes/cli`. Lanza una excepción controlada si Node o HyperFrames no están instalados.
- `playwright`: Fuerza la captura clásica frame a frame con Playwright Headless y pipe directo a FFmpeg `image2pipe`.

**Dependencias entre pasos del pipeline:**

| Paso | Comando / Módulo | Entrada | Salida | Depende de |
|---|---|---|---|---|
| 1 | `audio.voice_registry` (`Flujo 1`) | Muestra `.wav` (3–30 s) o preset ID | Fila en `voices` + `data/voices/<id>.wav` | — |
| 2 | `core.orchestrator.create_project` | Metadatos (`id`, `width`, `height`, `fps`) | Fila en `projects` + carpeta `workspaces/<id>/` | — |
| 3 | `audio.dialogue_synth` + `audio.aligner` (`Flujos 2 y 3`) | Guion por turnos + `voices` | `dialogue_master.wav` + `dialogue_lines` + `word_timestamps` | Pasos 1 y 2 |
| 4 | `scenes.generator` + `scenes.validator` (`Flujo 4`) | `word_timestamps` + código/prompt HTML | `scenes/scene_XX.html` con `validation_status='valid'` | Paso 3 |
| 5 | `core.orchestrator.audit_project_visuals` (`Flujo 6 — Check`) | Escenas validadas + composición `index.html` | Reporte de auditoría visual (80+ checks de HyperFrames o DOM Playwright) | Paso 4 |
| 6 | `render.engine` / `hyperframes_bridge` (`Flujo 6 — Render`) | Escenas validadas + `dialogue_master.wav` | `workspaces/<id>/output/final.mp4` (`60 FPS`) | Pasos 4 y 5 |

---

## 5. Despliegue

### 5C. Ejecución en Servidor Local / Contenedor Docker

```bash
# Construir imagen con Python 3.11, FFmpeg, Chromium Headless y Node.js opcional
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

| Problema / Síntoma | Causa probable | Solución / Comportamiento del Sistema |
|---|---|---|
| `Executable doesn't exist` al validar escena o renderizar | Falta instalar el binario de Chromium para Playwright | Ejecutar `uv run playwright install chromium` y verificar con `uv run python -m mcp_videos.cli doctor`. |
| `ffmpeg: command not found` en síntesis o renderizado | `ffmpeg` no está instalado o no se encuentra en el `PATH` | Instalar FFmpeg (`winget install Gyan.FFmpeg` en Windows, `sudo apt install ffmpeg` en Ubuntu, o `pacman -S ffmpeg`) y verificar con `doctor`. |
| Advertencia: *«El binario FFmpeg no incluye soporte libass (filtro 'ass')»* al renderizar con subtítulos | En Windows o entornos Conda/Anaconda, la compilación de FFmpeg suele carecer de la biblioteca `libass` (`--enable-libass`), por lo que no reconoce el filtro `-vf ass=...`. | `mcp-videos` detecta automáticamente la ausencia del filtro `ass` mediante `_ffmpeg_supports_ass_filter` (`ffmpeg -hide_banner -filters`), omite `-vf ass=...` para no romper el pipe de codificación y **mantiene los subtítulos sincronizados en pantalla a través del DOM HTML**. Para habilitar quemado por FFmpeg, instale una build completa con `winget install Gyan.FFmpeg`. |
| `HyperFramesNotFoundError` al ejecutar `check` o `render` con `--backend hyperframes` | Se forzó explícitamente el backend `hyperframes` pero el paquete no está instalado en el sistema ni en `MCP_VIDEOS_HYPERFRAMES_DIR`. | Instalar HyperFrames con `npm install -g hyperframes` o usar `--backend auto` para permitir el fallback automático al motor Playwright. |
| `SceneNotValidatedError` al intentar exportar a MP4 | Una escena falló la validación DOM/JS o el diálogo se regeneró tras guardar la escena alterando los timestamps. | Revisar `validation_errors_json` en SQLite o reejecutar `save-scene` para validar la escena frente a los nuevos timestamps de palabra. |
| Síntesis lenta en máquinas sin GPU NVIDIA | Se seleccionó un modelo pesado en precisión `float32` sobre CPU. | Configurar `LOCAL_INFERENCE_DEVICE=cpu` y `WHISPER_COMPUTE_TYPE=int8` en `.env`, y utilizar voces ligeras `kokoro`/`piper` para iteración rápida. |
| Puerto `8080` ocupado al abrir el Estudio Web | Otro servicio local está escuchando en el puerto predeterminado. | Definir `PREVIEW_PORT=8081` en `.env` o especificar `--port 8081` al arrancar el servidor. |
