from typing import Any

import orjson
from fastapi.responses import Response


ORJSON_RESPONSE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def dumps_json(content: Any) -> bytes:

    return orjson.dumps(content, option=ORJSON_RESPONSE_OPTIONS)


class SGLangORJSONResponse(Response):


    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return dumps_json(content)


def orjson_response(content: Any, status_code: int = 200) -> Response:

    return SGLangORJSONResponse(content=content, status_code=status_code)
