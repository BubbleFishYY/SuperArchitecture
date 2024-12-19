import importlib
from typing import List, Optional, Type

import torch.nn as nn

from TD_Pipe.logger import init_logger
from TD_Pipe.utils import is_hip

logger = init_logger(__name__)

_MODEL_CLASSES_SUPPORT_PIPELINE_PARALLEL = [
    "LlamaForCausalLM",
    "OPTForCausalLM",
    "MixtralForCausalLM"
]

# Architecture -> (module, class).
_MODELS = {
    "AquilaModel": ("aquila", "AquilaForCausalLM"),
    "AquilaForCausalLM": ("aquila", "AquilaForCausalLM"),  # AquilaChat2
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),  # baichuan-7b
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),  # baichuan-13b
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "InternLMForCausalLM": ("internlm", "InternLMForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("mistral", "MistralForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "PhiForCausalLM": ("phi_1_5", "PhiForCausalLM"),
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    "YiForCausalLM": ("yi", "YiForCausalLM"),
}

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_PARTIALLY_SUPPORTED_MODELS = {
    "MistralForCausalLM":
    "Sliding window attention is not yet supported in ROCm's flash attention",
    "MixtralForCausalLM":
    "Sliding window attention is not yet supported in ROCm's flash attention",
}


class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str, is_pipeline: bool = False) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None
        if is_pipeline and model_arch not in _MODEL_CLASSES_SUPPORT_PIPELINE_PARALLEL:
            logger.warning(
                f"Model architecture {model_arch} does not support pipeline parallelism.")
            return None
        
        if is_hip():
            if model_arch in _ROCM_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"Model architecture {model_arch} is not supported by "
                    "ROCm for now.")
            if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    f"Model architecture {model_arch} is partially supported "
                    "by ROCm: " + _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch])

        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"TD_Pipe.model_executor.models.{module_name}")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())


__all__ = [
    "ModelRegistry",
]
