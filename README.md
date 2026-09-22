# Agente local

Agente de programación que corre **100% en tu PC**: lee, busca, edita y ejecuta código en tu
proyecto usando modelos locales servidos con **llama.cpp** + **llama-swap**, con una GUI para
configurarlo todo.

- **Multimodelo**: roles `main` (el agente), `fast` (resúmenes) y `draft` (decodificación especulativa).
- **Contexto sin caídas**: compacta automáticamente y, si no hay forma, explica por qué y qué hacer.
- **Seguro**: solo accede al workspace; editar y ejecutar comandos requiere tu aprobación (con diff).
- **GUI**: requisitos explicados, catálogo de modelos con descarga, calculadora de VRAM, servidor,
  chat y una sección «Aprende».

## Instalación (Windows)

1. **Python 3.12**: `winget install Python.Python.3.12`
2. **llama.cpp** (build CUDA): en <https://github.com/ggml-org/llama.cpp/releases> descarga
   `llama-…-bin-win-cuda-12.x-x64.zip` y `cudart-llama-bin-win-cuda-12.x-x64.zip`; descomprime ambos
   en `bin/`.
3. **llama-swap**: en <https://github.com/mostlygeek/llama-swap/releases> descarga
   `llama-swap_…_windows_amd64.zip` y descomprímelo en `bin/`.
4. Arranca la GUI:

   ```powershell
   .\start.bat
   ```

   (o doble clic en `start.bat`). La primera vez crea `.venv` e instala las dependencias. Si la
   ventana nativa falla, usa `.\start.bat -Browser`. `start.bat` existe porque Windows bloquea los
   `.ps1` por defecto; ejecuta `start.ps1` saltándose esa política solo para ese proceso.

5. En la GUI: **Requisitos** (todo en verde) → **Modelos** (descarga y asigna roles) →
   **Configuración** (pulsa **Recalcular** para ajustar contexto, KV cache y capas en GPU a tu
   hardware; revisa la calculadora de VRAM) → **Servidor** (Arrancar) → **Chat**.

## Control de versiones

El repositorio solo contiene el código (~200 KB). `.gitignore` deja fuera todo lo que se descarga o
depende de la máquina:

- `bin/`, `models/`, `*.gguf`, zips e instaladores: se vuelven a descargar (pasos 2-3 y página Modelos).
- `.venv/`: lo recrea `start.bat`.
- `generated/`: `llama-swap.yaml` se regenera al guardar la configuración.
- `config/models.yaml`: tu configuración local. Si no existe, se crea con valores por defecto;
  después asigna los modelos y pulsa **Recalcular**.

## Uso por terminal

Con el servidor en marcha (desde la GUI):

```powershell
.venv\Scripts\python -m agent.cli -w D:\ruta\a\tu\proyecto
```

## Estructura

```
agent/core/     bucle, herramientas, cliente LLM, gestión de contexto (sin dependencias de UI)
agent/server/   requisitos, calculadora de VRAM, generación de llama-swap.yaml, gestor, descargas
agent/gui/      páginas NiceGUI
config/         models.yaml (tu configuración) y catalog.yaml (modelos recomendados)
generated/      llama-swap.yaml (se regenera al guardar; no editar)
bin/ models/    binarios y archivos .gguf (no se versionan)
```

## Tests

```powershell
.venv\Scripts\python -m pytest
```

No necesitan GPU ni modelos: el bucle se prueba con un LLM simulado, incluidos los casos de fallo
(desbordamiento de contexto, respuestas cortadas, servidor caído, JSON inválido).
