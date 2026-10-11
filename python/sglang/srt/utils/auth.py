from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    error_status_code: int = 401


class AuthLevel(str, Enum):


    NORMAL = "normal"
    ADMIN_OPTIONAL = "admin_optional"
    ADMIN_FORCE = "admin_force"


def auth_level(level: AuthLevel):


    def decorator(func):
        func._auth_level = level
        return func

    return decorator


def _get_auth_level_from_app_and_scope(app: Any, scope: dict) -> AuthLevel:


    from starlette.routing import Match


    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(
        app, "routes", []
    )

    for route in routes:
        try:
            match, child_scope = route.matches(scope)
        except Exception:
            continue
        if match == Match.FULL:
            endpoint = child_scope.get("endpoint") or getattr(route, "endpoint", None)
            level = getattr(endpoint, "_auth_level", None)
            return level if isinstance(level, AuthLevel) else AuthLevel.NORMAL

    return AuthLevel.NORMAL


def app_has_admin_force_endpoints(app: Any) -> bool:

    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(
        app, "routes", []
    )
    for route in routes:
        endpoint = getattr(route, "endpoint", None)
        if getattr(endpoint, "_auth_level", None) == AuthLevel.ADMIN_FORCE:
            return True
    return False


def decide_request_auth(
    *,
    method: str,
    path: str,
    authorization_header: Optional[str],
    api_key: Optional[str],
    admin_api_key: Optional[str],
    auth_level: AuthLevel,
) -> AuthDecision:

    if method == "OPTIONS":
        return AuthDecision(allowed=True)

    if path.startswith("/health") or path.startswith("/metrics"):
        return AuthDecision(allowed=True)

    def _check_bearer_token(
        authorization_header: Optional[str], expected_token: str
    ) -> bool:

        if not authorization_header:
            return False
        parts = authorization_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        return secrets.compare_digest(parts[1], expected_token)


    if auth_level == AuthLevel.ADMIN_FORCE:
        if not admin_api_key:
            return AuthDecision(allowed=False, error_status_code=403)
        if not _check_bearer_token(authorization_header, admin_api_key):
            return AuthDecision(allowed=False)
        return AuthDecision(allowed=True)


    if auth_level == AuthLevel.ADMIN_OPTIONAL:
        if admin_api_key:
            return AuthDecision(
                allowed=_check_bearer_token(authorization_header, admin_api_key)
            )
        elif api_key:
            return AuthDecision(
                allowed=_check_bearer_token(authorization_header, api_key)
            )
        else:
            return AuthDecision(allowed=True)


    if api_key:
        return AuthDecision(allowed=_check_bearer_token(authorization_header, api_key))

    return AuthDecision(allowed=True)


def add_api_key_middleware(
    app,
    *,
    api_key: Optional[str],
    admin_api_key: Optional[str],
):


    from fastapi.responses import ORJSONResponse
    from starlette.requests import Request

    class _ApiKeyASGIMiddleware:


        def __init__(self, app, *, api_key, admin_api_key, fastapi_app):
            self.app = app
            self.api_key = api_key
            self.admin_api_key = admin_api_key
            self.fastapi_app = fastapi_app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            request = Request(scope, receive=receive)
            path = request.url.path
            authz = request.headers.get("Authorization")
            level = _get_auth_level_from_app_and_scope(self.fastapi_app, scope)
            decision = decide_request_auth(
                method=request.method,
                path=path,
                authorization_header=authz,
                api_key=self.api_key,
                admin_api_key=self.admin_api_key,
                auth_level=level,
            )

            if not decision.allowed:
                response = ORJSONResponse(
                    content={
                        "error": (
                            "Unauthorized"
                            if decision.error_status_code == 401
                            else "Forbidden"
                        )
                    },
                    status_code=decision.error_status_code,
                )
                await response(scope, receive, send)
                return

            await self.app(scope, receive, send)

    app.add_middleware(
        _ApiKeyASGIMiddleware,
        api_key=api_key,
        admin_api_key=admin_api_key,
        fastapi_app=app,
    )
