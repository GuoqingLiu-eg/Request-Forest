from __future__ import annotations

from starlette.requests import Request


class _SentResponse:


    def __init__(self, status_code: int):
        self.status_code = status_code


class _PureASGIDispatch:


    def __init__(self, app, dispatch):
        self.app = app
        self.dispatch = dispatch

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        status_code = 500

        async def call_next(_request):
            nonlocal status_code

            async def send_and_capture(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await self.app(scope, receive, send_and_capture)
            return _SentResponse(status_code)

        await self.dispatch(request, call_next)


def patch_app_http_middleware(app):

    _orig = app.middleware

    def _fixed(middleware_type):
        if middleware_type == "http":

            def decorator(fn):
                app.add_middleware(_PureASGIDispatch, dispatch=fn)
                return fn

            return decorator
        return _orig(middleware_type)

    app.middleware = _fixed
