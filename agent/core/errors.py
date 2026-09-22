"""Errores con explicación para el usuario. El bucle los convierte en eventos AgentError."""

from __future__ import annotations


class AgentFailure(Exception):
    """Fallo explicable: título, causa, sugerencias y una acción opcional para la GUI."""

    def __init__(
        self,
        title: str,
        cause: str,
        suggestions: list[str] | None = None,
        detail: str = "",
        action: str | None = None,
    ):
        super().__init__(f"{title}: {cause}")
        self.title = title
        self.cause = cause
        self.suggestions = suggestions or []
        self.detail = detail
        self.action = action


class ServerUnavailable(AgentFailure):
    def __init__(self, endpoint: str, detail: str = ""):
        super().__init__(
            "El servidor de modelos no responde",
            f"No hay nada escuchando en {endpoint}.",
            ["Arranca el servidor en la página «Servidor».",
             "Si lo arrancaste a mano, revisa que el puerto coincida con el de Configuración."],
            detail,
            action="server",
        )


class ModelLoadFailed(AgentFailure):
    def __init__(self, model: str, detail: str = "", out_of_memory: bool = False):
        if out_of_memory:
            cause = (f"El modelo «{model}» no cabe en la memoria de la GPU (VRAM) "
                     "con el contexto configurado.")
            suggestions = [
                "Baja el contexto del modelo o usa KV cache q8_0/q4_0 en Configuración "
                "(la calculadora de VRAM te dice cuánto ocupa).",
                "Desactiva «mantener main y fast cargados» para no tener dos modelos en VRAM.",
                "Cierra otros programas que usen la GPU (juegos, navegadores con aceleración).",
            ]
        else:
            cause = f"llama-server no pudo cargar «{model}»."
            suggestions = [
                "Revisa el registro en la página «Servidor».",
                "Comprueba que el archivo .gguf existe y no está incompleto.",
            ]
        super().__init__("No se pudo cargar el modelo", cause, suggestions, detail, action="server")


class ContextExhausted(AgentFailure):
    """No hay forma de meter la petición en la ventana de contexto."""

    def __init__(self, breakdown: dict[str, int], limit: int, budget: int, hint: str | None = None):
        parts = ", ".join(f"{k}: {v:,}".replace(",", ".") for k, v in breakdown.items())
        cause = (f"La conversación necesita unos {sum(breakdown.values()):,} tokens y el modelo "
                 f"solo admite {budget:,} de entrada (contexto {limit:,} menos la reserva "
                 f"para la respuesta). Desglose: {parts}.").replace(",", ".")
        suggestions = []
        if hint:
            suggestions.append(hint)
        suggestions += [
            "Empieza una conversación nueva (lo ya hecho en disco se conserva).",
            "Pide leer archivos por rangos de líneas en lugar de enteros.",
            "Divide la tarea en pasos más pequeños.",
        ]
        super().__init__("Sin espacio en el contexto", cause, suggestions, action="new_chat")
        self.breakdown = breakdown
        self.limit = limit
        self.budget = budget


class ContextOverflowFromServer(Exception):
    """El servidor rechazó la petición por exceder el contexto (uso interno)."""
