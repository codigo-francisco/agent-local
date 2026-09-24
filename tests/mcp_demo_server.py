"""Servidor MCP mínimo (stdio) para los tests."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("demo")


@server.tool()
def sumar(a: int, b: int) -> str:
    """Suma dos números enteros."""
    return str(a + b)


@server.tool()
def fallar() -> str:
    """Siempre falla (para probar errores)."""
    raise ValueError("fallo a propósito")


if __name__ == "__main__":
    server.run("stdio")
