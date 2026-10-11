import logging

import jinja2
import transformers.utils.chat_template_utils as hf_chat_utils

from sglang.srt.utils import ImageData

logger = logging.getLogger(__name__)


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:

    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:

    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str = None,
) -> bool:

    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _try_extract_ast(chat_template: str):

    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception as e:
        logger.debug(f"Error when compiling Jinja template: {e}")
        return None


def detect_jinja_template_content_format(chat_template: str) -> str:


    if any(
        keyword in chat_template for keyword in ["image", "audio", "video", "vision"]
    ):
        return "openai"

    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return "string"

    try:

        for loop_ast in jinja_ast.find_all(jinja2.nodes.For):
            loop_iter = loop_ast.iter


            if _is_var_or_elems_access(loop_iter, "message", "content"):
                return "openai"


            if _is_var_or_elems_access(
                loop_iter, "msg", "content"
            ) or _is_var_or_elems_access(loop_iter, "m", "content"):
                return "openai"

        return "string"
    except Exception as e:
        logger.debug(f"Error when parsing AST of Jinja template: {e}")
        return "string"


def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    video_data: list,
    audio_data: list,
    modalities: list,
    use_dpsk_v32_encoding: bool = False,
) -> dict:

    if not isinstance(msg_dict.get("content"), list):

        return {k: v for k, v in msg_dict.items() if v is not None}

    if content_format == "openai" or use_dpsk_v32_encoding:


        processed_content_parts = []
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")

                if chunk_type in ("image_url", "input_image"):
                    image_obj = chunk.get("image_url") or {}
                    if isinstance(image_obj, str):
                        image_obj = {"url": image_obj, "detail": chunk.get("detail")}
                    mdp = image_obj.get("max_dynamic_patch", None)

                    image_data.append(
                        ImageData(
                            url=image_obj["url"],
                            detail=image_obj.get("detail") or "auto",
                            max_dynamic_patch=mdp,
                            content_hash=image_obj.get("content_hash"),
                        )
                    )

                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))

                    processed_content_parts.append({"type": "image"})
                elif chunk_type == "video_url":
                    video_obj = chunk.get("video_url") or {}
                    mdp = video_obj.get("max_dynamic_patch", None)
                    if mdp is None:
                        video_data.append(chunk["video_url"]["url"])
                    else:

                        video_data.append(
                            {
                                "url": video_obj["url"],
                                "max_dynamic_patch": mdp,
                            }
                        )
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))

                    processed_content_parts.append({"type": "video"})
                elif chunk_type == "audio_url":
                    audio_data.append(chunk["audio_url"]["url"])

                    processed_content_parts.append({"type": "audio"})
                elif chunk_type in ("text", "input_text"):

                    if use_dpsk_v32_encoding:
                        text_parts.append(chunk["text"])
                    else:

                        processed_content_parts.append(
                            {"type": "text", "text": chunk["text"]}
                        )
                elif chunk_type == "tool_reference":


                    processed_content_parts.append(chunk)

        new_msg = {
            k: v for k, v in msg_dict.items() if v is not None and k != "content"
        }
        if use_dpsk_v32_encoding:
            new_msg["content"] = " ".join(text_parts) if text_parts else ""
        else:
            new_msg["content"] = processed_content_parts
        return new_msg

    elif content_format == "string":

        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict) and chunk.get("type") in ("text", "input_text"):
                text_parts.append(chunk["text"])


        new_msg = msg_dict.copy()
        new_msg["content"] = " ".join(text_parts) if text_parts else ""
        new_msg = {k: v for k, v in new_msg.items() if v is not None}
        return new_msg

    else:
        raise ValueError(f"Invalid content format: {content_format}")
