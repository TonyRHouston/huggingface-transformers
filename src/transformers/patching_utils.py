# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import re
from contextlib import contextmanager

import torch.nn as nn

from .core_model_loading import WeightConverter


class Patcher:
    """
    Temporarily replaces model components during loading to restructure models for specific requirements.

    Use model patching to restructure models when [`AttentionInterface`], [`ExpertsInterface`], or the kernels
    registry don't support your use case. Common scenarios include quantization library compatibility,
    structural optimizations, and architectural experimentation.

    Args:
        class_mapping (`Dict[str, type[nn.Module]]`):
            Mapping from original class names to replacement classes. Class names must exactly match those
            in the model's module (e.g., `"Qwen2MoeExperts"` → `ModuleListExperts`).
        filtered_weight_conversion_patterns (`str` or `List[str]`, *optional*):
            Regex patterns to exclude weight conversions. Any conversion with source or target patterns
            matching these will be filtered out. Use this when structural changes make existing conversions
            incompatible (e.g., `[".gate_up_proj", ".down_proj"]` when changing expert structure).
        extra_weight_conversions (`WeightConverter` or `List[WeightConverter]`, *optional*):
            Additional weight conversions for the new structure. These are prepended to existing conversions,
            allowing you to transform weights to match your modified architecture (e.g., concatenating
            separate Q, K, V weights into a fused QKV projection).

    Example:
        ```python
        from transformers import AutoModelForCausalLM, Patcher, WeightConverter, Concatenate

        # Restructure MoE experts for quantization compatibility
        patcher = Patcher(
            class_mapping={"Qwen2MoeExperts": SequentialExperts},
            filtered_weight_conversion_patterns=[".gate_up_proj", ".down_proj"],
        )

        # Fuse QKV projections for efficient quantization kernels
        patcher = Patcher(
            class_mapping={"Qwen2MoeAttention": FusedQKVAttention},
            extra_weight_conversions=[
                WeightConverter(
                    source_patterns=["q_proj.weight", "k_proj.weight", "v_proj.weight"],
                    target_patterns=["qkv_proj.weight"],
                    operations=[Concatenate(dim=0)],
                ),
                WeightConverter(
                    source_patterns=["q_proj.bias", "k_proj.bias", "v_proj.bias"],
                    target_patterns=["qkv_proj.bias"],
                    operations=[Concatenate(dim=0)],
                ),
            ],
        )

        model = AutoModelForCausalLM.from_pretrained("model-name", patcher=patcher)
        ```
    """

    def __init__(
        self,
        class_mapping: dict[str, type[nn.Module]],
        filtered_weight_conversion_patterns: str | list[str] | None = None,
        extra_weight_conversions: WeightConverter | list[WeightConverter] | None = None,
    ):
        self.class_mapping = class_mapping
        self.filtered_patterns = (
            [filtered_weight_conversion_patterns]
            if isinstance(filtered_weight_conversion_patterns, str)
            else filtered_weight_conversion_patterns
        )
        self.extra_conversions = (
            [extra_weight_conversions]
            if isinstance(extra_weight_conversions, WeightConverter)
            else extra_weight_conversions
        )
        self._original_classes = {}

    @contextmanager
    def get_patching_context(self, model_class: type[nn.Module]):
        """
        Context manager to temporarily patch the model class.
        The specified classes in class_mapping will be replaced with the new classes for
        the duration of the context, and then restored to their original state afterwards.

        Args:
            model_class (`type[nn.Module]`):
                The model class to patch (e.g., Qwen2MoeForCausalLM).
        """
        modeling_module = importlib.import_module(model_class.__module__)

        try:
            for module_name, replacement_class in self.class_mapping.items():
                if hasattr(modeling_module, module_name):
                    self._original_classes[module_name] = getattr(modeling_module, module_name)
                    setattr(modeling_module, module_name, replacement_class)
                else:
                    raise AttributeError(
                        f"Module '{modeling_module.__name__}' does not have a class named '{module_name}' to patch."
                    )

            yield

        finally:
            for module_name, original_class in self._original_classes.items():
                setattr(modeling_module, module_name, original_class)
            self._original_classes.clear()

    def update_weight_conversions(self, weight_conversions: list[WeightConverter]) -> list[WeightConverter]:
        """
        Filter and add weight conversions according to the patcher configuration.

        Args:
            weight_conversions (`List[WeightConverter]`):
                The list of weight conversions to process.

        Returns:
            `List[WeightConverter]`: The processed list of weight conversions.
        """
        filtered = self._filter_conversions(weight_conversions)
        return self._add_conversions(filtered)

    def _filter_conversions(self, weight_conversions: list[WeightConverter]) -> list[WeightConverter]:
        """Filter out weight conversions that match any of the specified patterns."""
        if self.filtered_patterns is None:
            return weight_conversions

        filtered = []
        for conversion in weight_conversions:
            conversion_patterns = conversion.source_patterns + conversion.target_patterns
            if any(
                any(re.search(pattern, conv_pattern) for conv_pattern in conversion_patterns)
                for pattern in self.filtered_patterns
            ):
                continue
            filtered.append(conversion)

        return filtered

    def _add_conversions(self, weight_conversions: list[WeightConverter]) -> list[WeightConverter]:
        """Add extra weight conversions specified in the patcher configuration."""
        if self.extra_conversions is None:
            return weight_conversions

        return self.extra_conversions + weight_conversions
