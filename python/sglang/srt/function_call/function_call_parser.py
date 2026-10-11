import inspect
import logging
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union

from sglang.srt.entrypoints.openai.protocol import (
    LegacyStructuralTagResponseFormat,
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolCallConstraint,
    ToolChoice,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.dots_detector import DotsToolDetector
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.minimax_m3 import MinimaxM3Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.muse_glimmer_detector import MuseGlimmerDetector
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.spark25_detector import Spark25Detector
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.srt.function_call.utils import (
    _get_tool_schema_defs,
    get_json_schema_constraint,
)

logger = logging.getLogger(__name__)


class FunctionCallParser:


    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "apertus2509": Apertus2509Detector,
        "cohere_command4": CohereCommand4Detector,
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseekv32": DeepSeekV32Detector,
        "deepseekv4": DeepSeekV4Detector,
        "dots": DotsToolDetector,
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm47MoeDetector,
        "gpt-oss": GptOssDetector,
        "kimi_k2": KimiK2Detector,
        "kimi_k3": KimiK3Detector,
        "lfm2": Lfm2Detector,
        "llama3": Llama32Detector,
        "mimo": MiMoDetector,
        "minicpm5": MiniCPM5Detector,
        "mistral": MistralDetector,
        "muse": MuseGlimmerDetector,
        "poolside_v1": PoolsideV1Detector,
        "pythonic": PythonicDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "spark25": Spark25Detector,
        "step3": Step3Detector,
        "step3p5": Qwen3CoderDetector,
        "minimax-m2": MinimaxM2Detector,
        "minimax-m3": MinimaxM3Detector,
        "trinity": TrinityDetector,
        "interns1": InternlmDetector,
        "hermes": HermesDetector,
        "hunyuan": HunyuanDetector,
        "gigachat3": GigaChat3Detector,
        "gemma4": Gemma4Detector,
        "inkling": InklingDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str, tokenizer=None):
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            kwargs = {}
            if tokenizer is not None:
                sig = inspect.signature(detector_class)
                if "tokenizer" in sig.parameters:
                    kwargs["tokenizer"] = tokenizer
            detector = detector_class(**kwargs)
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools
        self.tool_strict_level = envs.SGLANG_TOOL_STRICT_LEVEL.get()

    def has_tool_call(self, text: str) -> bool:

        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:

        if not self.tools:
            return full_text, []
        has_tool_call = self.detector.has_tool_call(full_text)
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list or has_tool_call:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:

        if not self.tools:
            return chunk_text, []
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def parse_stream_end(self) -> Tuple[str, list[ToolCallItem]]:

        if not self.tools:
            return "", []
        sp_result = self.detector.finish(self.tools)
        return sp_result.normal_text, sp_result.calls

    def get_legacy_structural_tag(
        self, at_least_one: bool = False
    ) -> StructuralTagResponseFormat:


        _get_tool_schema_defs(self.tools)

        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)


            is_strict = (
                function.strict or self.tool_strict_level >= ToolStrictLevel.PARAMETER
            )
            schema = function.parameters if is_strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema or {},  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)


        return LegacyStructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
            at_least_one=at_least_one,
        )

    def get_structure_constraint(
        self,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]],
        parallel_tool_calls: bool = True,
        thinking_mode: bool = False,
    ) -> Optional[ToolCallConstraint]:

        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)
        should_constrain_auto = tool_choice == "auto" and (
            any(tool.function.strict for tool in self.tools)
            or self.tool_strict_level >= ToolStrictLevel.FUNCTION
        )


        try:
            if tool_choice == "auto" and not should_constrain_auto:
                structural_tag = self.detector.get_auto_tool_call_structural_tag(
                    tools=self.tools,
                    thinking_mode=thinking_mode,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if structural_tag is not None:
                    return ("structural_tag", structural_tag)

            if is_required or should_constrain_auto:
                structural_tag_tools = self.tools
                if self.tool_strict_level >= ToolStrictLevel.PARAMETER:
                    structural_tag_tools = [
                        tool.model_copy(
                            update={
                                "function": tool.function.model_copy(
                                    update={"strict": True}
                                )
                            }
                        )
                        for tool in self.tools
                    ]
                structural_tag = self.detector.get_structural_tag(
                    tools=structural_tag_tools,
                    thinking_mode=thinking_mode,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if structural_tag is not None:
                    return ("structural_tag", structural_tag)


                if self.detector.supports_structural_tag():


                    tag = self.get_legacy_structural_tag(at_least_one=is_required)
                    return ("structural_tag", tag)

            if (
                tool_choice == "required" or isinstance(tool_choice, ToolChoice)
            ) and not self.detector.parses_required_natively():
                json_schema = get_json_schema_constraint(
                    self.tools, tool_choice, parallel_tool_calls=parallel_tool_calls
                )
                return ("json_schema", json_schema)
        except Exception as e:
            logger.error(f"Error getting structure constraint: {e}")
            return None
