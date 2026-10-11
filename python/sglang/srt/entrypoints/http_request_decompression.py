import asyncio
import io
import logging

import zstandard
from fastapi.responses import Response
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


def _zstd_decompress(raw: bytes) -> bytes:
    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()


_DECOMPRESSORS = {"zstd": _zstd_decompress}


def _rewrite_headers(headers, new_len):

    out = [
        (k, v)
        for (k, v) in headers
        if k not in (b"content-length", b"x-body-compressed")
    ]
    out.append((b"content-length", str(new_len).encode()))
    return out


class RequestDecompressionMiddleware:


    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):

        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        method = Headers(scope=scope).get("x-body-compressed")
        if method is None:
            return await self.app(scope, receive, send)


        decompress = _DECOMPRESSORS.get(method)
        if decompress is None:
            return await Response(
                f"unsupported x-body-compressed {method!r}; "
                f"supported: {sorted(_DECOMPRESSORS)}",
                status_code=400,
            )(scope, receive, send)


        body = b""
        more_body = True
        while more_body:
            message = await receive()

            if message["type"] != "http.request":
                return await self.app(scope, receive, send)
            body += message.get("body", b"")
            more_body = message.get("more_body", False)


        try:
            loop = asyncio.get_running_loop()
            body = await loop.run_in_executor(None, decompress, body)
        except Exception as e:
            logger.warning("request body decompress failed: %s", e)
            return await Response("decompress failed", status_code=400)(
                scope, receive, send
            )


        scope = dict(scope)
        scope["headers"] = _rewrite_headers(scope["headers"], len(body))


        body_sent = False

        async def wrapped_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, wrapped_receive, send)
