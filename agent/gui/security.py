"""Solo esta máquina puede manejar la GUI (que ejecuta comandos).

Escuchar en 127.0.0.1 no basta: NiceGUI acepta websockets de cualquier origen
(`cors_allowed_origins='*'`) y, con *DNS rebinding*, una web abierta en el navegador puede hacer que
su dominio apunte a 127.0.0.1 y hablar con la GUI como si fuera ella. Se rechaza toda petición cuyo
`Host` no sea esta máquina o cuyo `Origin` sea otra web.
"""

from __future__ import annotations

import logging

log = logging.getLogger("agent.gui")
LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]")


class LocalOnly:
    """Middleware ASGI (http y websocket)."""

    def __init__(self, app, port: int):
        self.app = app
        self.hosts = {f"{name}:{port}" for name in LOCAL_NAMES}
        self.origins = {f"http://{h}" for h in self.hosts}

    def allowed(self, scope) -> bool:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1").lower()
                   for k, v in scope.get("headers") or []}
        host, origin = headers.get("host", ""), headers.get("origin")
        return host in self.hosts and (origin is None or origin in self.origins)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or self.allowed(scope):
            await self.app(scope, receive, send)
            return
        log.warning("Petición rechazada (%s %s): Host/Origin ajenos", scope["type"], scope.get("path"))
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
            await send({"type": "http.response.body",
                        "body": "Solo se puede usar desde esta máquina.".encode()})
        else:
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 1008})
