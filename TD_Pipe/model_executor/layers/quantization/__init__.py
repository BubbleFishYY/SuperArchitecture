from typing import Type

from TD_Pipe.model_executor.layers.quantization.base_config import QuantizationConfig
from TD_Pipe.model_executor.layers.quantization.awq import AWQConfig
from TD_Pipe.model_executor.layers.quantization.gptq import GPTQConfig
from TD_Pipe.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig

_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]
