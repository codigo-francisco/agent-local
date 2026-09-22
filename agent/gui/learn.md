## ¿Cómo funciona el agente?
Un agente es un **bucle**: el modelo lee tu petición y decide usar una **herramienta** (leer un archivo,
buscar, editar, ejecutar un comando). La aplicación ejecuta esa herramienta, le devuelve el resultado y
el modelo decide el siguiente paso. Así hasta que da la tarea por terminada.

```
tú: "arregla el test que falla"
 → modelo: run_command("pytest")        → resultado: 1 test falla en calc.py
 → modelo: read_file("calc.py")         → resultado: el código
 → modelo: edit_file(...)               → tú apruebas el diff
 → modelo: run_command("pytest")        → todo verde
 → modelo: "Listo: el error era..."
```

El modelo **nunca toca tu disco directamente**: solo pide herramientas, y la app las ejecuta
dentro de la carpeta del proyecto. Editar archivos y ejecutar comandos requiere tu aprobación
(salvo que actives el modo automático).

## ¿Qué es un archivo GGUF?
Es el formato de llama.cpp para guardar un modelo: los **pesos** de la red neuronal más los metadatos
(arquitectura, vocabulario, plantilla de chat). Un solo archivo contiene todo lo necesario.
Los encontrarás en Hugging Face, normalmente en repos de *bartowski*, *unsloth* o del propio fabricante.

## Parámetros: 3B, 7B, 14B…
La "B" son miles de millones de parámetros. Más parámetros = más capacidad de razonar y de seguir
instrucciones complejas, pero también más memoria y más lentitud.
Para un agente que usa herramientas, **14B es el mínimo recomendable** para trabajo serio; los de 7B
funcionan para tareas sencillas y los de 3B o menos solo para tareas auxiliares.

## Cuantización: Q4_K_M, Q8_0…
Los pesos originales usan 16 bits por número. **Cuantizar** los comprime a menos bits:

| Cuantización | Bits aprox. | Tamaño 14B | Calidad |
|---|---|---|---|
| F16 | 16 | ~28 GB | original |
| Q8_0 | 8,5 | ~15 GB | prácticamente idéntica |
| Q6_K | 6,6 | ~12 GB | excelente |
| **Q4_K_M** | 4,8 | **~9 GB** | **muy buena: el punto dulce** |
| Q3_K_M | 3,9 | ~7 GB | se nota la pérdida |

Regla práctica: **un modelo más grande en Q4 suele ser mejor que uno más pequeño en Q8**.

## Contexto y ventana de contexto
El **contexto** es todo lo que el modelo "ve" a la vez: instrucciones del sistema, definiciones de
herramientas, la conversación, los archivos leídos y los resultados de comandos. Se mide en **tokens**
(≈ 3-4 caracteres de código cada uno).

Parte de la ventana se **reserva para la respuesta** (ajuste «tokens de salida»); el resto es lo
que el medidor del chat llama *tokens útiles*.

## ¿Qué pasa cuando el contexto se llena?
El agente **no se corta ni falla sin explicación**. Antes de cada llamada comprueba si todo cabe y, si
no, compacta en este orden, avisándote en el chat:

1. **Omite resultados antiguos de herramientas** (el modelo puede volver a pedirlos).
2. **Resume los mensajes más antiguos** con el modelo *fast* y guarda el resumen en las instrucciones.
3. **Recorta por el medio** los mensajes enormes, dejando una nota para que el modelo lea por rangos.
4. Si aun así no cabe, **no llama al modelo** y te explica el porqué con un desglose (sistema,
   herramientas, historial, último mensaje) y qué puedes hacer: subir el contexto (con su coste en
   VRAM), empezar una conversación nueva o dividir la tarea.

Si el servidor rechaza una petición por contexto (las estimaciones pueden fallar), compacta más y
reintenta una vez. Si una respuesta se corta por el límite de salida, le pide que continúe; si lo que
se cortó fue una llamada a herramienta, **no la ejecuta** a medias y le pide que la divida.

## KV cache: por qué el contexto cuesta VRAM
Para no recalcular todo en cada token, el modelo guarda en la GPU unas matrices por cada token del
contexto: la **KV cache**. Su tamaño crece linealmente con el contexto:

`KV = capas × cabezas_kv × (dim_clave + dim_valor) × contexto × bytes`

Para Qwen2.5-Coder-14B: ~192 KB por token en f16 → **32K tokens = 6 GB**. Con **q8_0** la mitad
(3 GB) y apenas se nota en calidad; con q4_0, un cuarto, con algo más de pérdida. Por eso q8_0 es el
valor por defecto. La cuantización de la KV cache necesita *flash attention* activada.

## VRAM: el recurso que manda
Todo lo que el modelo usa debe caber en la memoria de la GPU (VRAM): **pesos + KV cache + ~0,6 GB
de buffers**. Si no cabe, llama.cpp falla al cargar ("out of memory") o, si bajas las *capas en GPU*,
deja parte en la RAM y va mucho más lento.

La **calculadora de Configuración** hace estas cuentas con los datos reales del archivo GGUF.
Con 16 GB: un 14B Q4 (≈8,4 GB) + 24K de contexto q8_0 (≈2,4 GB) + margen (0,6 GB) + un 3B para
resúmenes con su contexto (≈2,7 GB) ≈ 14 GB, dejando algo de aire para Windows.

## Roles: main, fast y draft
- **main**: el agente en sí. El modelo más capaz que quepa, con buen *tool calling*.
- **fast**: tareas auxiliares baratas, sobre todo **resumir** la conversación cuando el contexto se
  llena. Si no hay, lo hace el principal (más lento).
- **draft** (opcional): **decodificación especulativa**. Un modelo diminuto de la misma familia propone
  varios tokens y el principal los verifica todos de una pasada: si acierta, generas 1,5-2× más rápido
  con exactamente la misma calidad. Solo funciona si ambos comparten vocabulario (misma familia).

## llama.cpp, llama-server y llama-swap
- **llama.cpp**: el motor que ejecuta modelos GGUF en CPU/GPU. Es la base de Ollama, LM Studio y otros.
- **llama-server**: el programa de llama.cpp que carga *un* modelo y ofrece una API compatible con la
  de OpenAI (`/v1/chat/completions`), incluido *tool calling* (con `--jinja`).
- **llama-swap**: un proxy que da **una sola dirección para varios modelos**. Según el campo `model` de
  cada petición arranca el llama-server adecuado. Los modelos de un mismo *grupo* pueden estar cargados
  a la vez (así main y fast no se turnan).

Como todo habla el protocolo de OpenAI, puedes apuntar el *endpoint* a Ollama o LM Studio sin cambiar
el agente.

## Tool calling (uso de herramientas)
El modelo debe haber sido entrenado para responder con llamadas estructuradas (JSON) a funciones. Si
un modelo no lo soporta bien, el agente no podrá leer ni editar archivos. Los Qwen2.5-Coder, Qwen3,
Llama 3.1+ y Mistral recientes lo soportan. Si un modelo escribe la llamada como texto
(`<tool_call>…`), el agente intenta rescatarla igualmente.

## Cómo elegir modelo para tu GPU
| VRAM | Principal recomendado | Contexto razonable |
|---|---|---|
| 8 GB | 7B Q4_K_M | 16K |
| 12 GB | 14B Q4_K_M (justo) o 7B Q6 | 8-16K / 32K |
| **16 GB** | **14B Q4_K_M** | **24-32K** |
| 24 GB | 32B Q4_K_M o 14B Q8 | 32K |

Los modelos **MoE** (p. ej. Qwen3-Coder-30B-A3B) son una excepción interesante: muchos parámetros pero
pocos activos, así que se pueden repartir entre GPU y RAM (`--n-cpu-moe`) con buena velocidad.

## Seguridad
- El agente solo accede a archivos **dentro del workspace**; cualquier ruta fuera se rechaza.
- Editar, crear archivos y ejecutar comandos requiere tu aprobación, con el diff o el comando exacto.
- Todo corre en local: tu código **no sale de tu PC**.
- Aun así, un comando aprobado puede hacer cualquier cosa que tú podrías hacer: lee antes de aprobar,
  y usa git para poder deshacer cambios.
