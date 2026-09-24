"""La GUI solo responde a esta máquina (defensa contra DNS rebinding)."""

import asyncio

import httpx

from agent.gui.security import LocalOnly


async def inner(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def get(headers):
    async def go():
        transport = httpx.ASGITransport(app=LocalOnly(inner, port=8765))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as c:
            return (await c.get("/", headers=headers)).status_code
    return asyncio.run(go())


def test_local_requests_pass():
    assert get({"Host": "127.0.0.1:8765"}) == 200
    assert get({"Host": "localhost:8765", "Origin": "http://localhost:8765"}) == 200


def test_foreign_host_or_origin_is_rejected():
    assert get({"Host": "evil.example:8765"}) == 403  # DNS rebinding
    assert get({"Host": "127.0.0.1:8765", "Origin": "https://evil.example"}) == 403
    assert get({"Host": "127.0.0.1:9999"}) == 403


def test_websocket_from_other_origin_is_closed():
    sent = []

    async def go():
        scope = {"type": "websocket", "path": "/_nicegui_ws/",
                 "headers": [(b"host", b"127.0.0.1:8765"), (b"origin", b"https://evil.example")]}

        async def receive():
            return {"type": "websocket.connect"}

        async def send(msg):
            sent.append(msg)

        await LocalOnly(inner, port=8765)(scope, receive, send)

    asyncio.run(go())
    assert sent == [{"type": "websocket.close", "code": 1008}]
