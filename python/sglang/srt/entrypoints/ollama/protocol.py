from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class OllamaMessage(BaseModel):


    role: str
    content: str
    images: Optional[List[str]] = None


class OllamaChatRequest(BaseModel):


    model: str
    messages: List[OllamaMessage]
    stream: bool = True
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[float, str]] = None
    think: Optional[Union[bool, Literal["low", "medium", "high"]]] = None


class OllamaChatResponse(BaseModel):


    model: str
    created_at: str
    message: OllamaMessage
    done: bool = True
    done_reason: Optional[str] = "stop"
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaChatStreamResponse(BaseModel):


    model: str
    created_at: str
    message: OllamaMessage
    done: bool = False
    done_reason: Optional[str] = None


class OllamaGenerateRequest(BaseModel):


    model: str
    prompt: str
    suffix: Optional[str] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[float, str]] = None
    images: Optional[List[str]] = None
    think: Optional[bool] = None


class OllamaGenerateResponse(BaseModel):


    model: str
    created_at: str
    response: str
    done: bool = True
    done_reason: Optional[str] = "stop"
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaGenerateStreamResponse(BaseModel):


    model: str
    created_at: str
    response: str
    done: bool = False
    done_reason: Optional[str] = None


class OllamaModelInfo(BaseModel):


    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Optional[Dict[str, Any]] = None


class OllamaTagsResponse(BaseModel):


    models: List[OllamaModelInfo]


class OllamaShowRequest(BaseModel):


    model: str


class OllamaShowResponse(BaseModel):


    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    modified_at: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
