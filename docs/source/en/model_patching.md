<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Model patching

Model patching allows you to temporarily replace model components during loading without modifying the original model code. This enables you to restructure models for specific requirements like quantization compatibility, apply optimizations, or experiment with architectural variants.

> [!WARNING]
> **Model patching should be used as a last resort** when you need to change the layout and structure of a module. For most customization needs, Transformers already provides better alternatives:
>
> - **[Attention customization](./attention_interface)** - Use the `AttentionInterface` to register optimized attention implementations without changing module structure. Simply set `attn_implementation="your_implementation"` in `from_pretrained`.
> - **[Experts customization](./experts_interface)** - Use the `ExpertsInterface` to register optimized MoE expert implementations. Set `experts_implementation="your_implementation"` in `from_pretrained`.
> - **[Kernels registry](./kernel_doc/overview)** - Register custom kernels for specific operations that are automatically dispatched during forward passes.
>
> Only use model patching when these interfaces don't support your use case—typically when you need structural changes that can't be achieved through custom forward implementations alone.

## When to use model patching

Model patching is designed for scenarios where you need to:

1. **Restructure for library compatibility** - Quantization libraries or other tools may require specific module layouts (e.g., sequential experts, fused projections)
2. **Apply structural optimizations** - Change module organization for performance benefits (e.g., fusing operations, changing expert routing)
3. **Experiment with variants** - Test architectural changes without modifying model source code

## Quick example

Here's a complete example restructuring a Mixture-of-Experts (MoE) model. While this example focuses on quantization compatibility, the same approach works for other optimizations:

```python
from transformers import AutoModelForCausalLM, Patcher, WeightConverter, Concatenate
import torch.nn as nn

# Define restructured expert module for quantization compatibility
class SequentialExperts(nn.ModuleList):
    """MoE experts as a ModuleList instead of custom container.
    
    Many quantization libraries prefer experts as a sequence of standard
    PyTorch linear layers for easier quantization application.
    """
    def __init__(self, config):
        super().__init__()
        for _ in range(config.num_experts):
            self.append(ExpertMLP(config))
        # ... rest of implementation

# Define fused attention for compute-intensive quantization kernels
class FusedQKVAttention(nn.Module):
    """Attention with fused Q, K, V projections.
    
    Fusing QKV projections into a single Linear layer enables
    compute-intensive quantization kernels that are more efficient.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.qkv_proj = nn.Linear(
            config.hidden_size, 
            3 * config.num_attention_heads * config.head_dim, 
            bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False
        )
        # ... rest of implementation

# Create patcher with both class replacements and weight conversions
patcher = Patcher(
    class_mapping={
        "Qwen2MoeExperts": SequentialExperts,
        "Qwen2MoeAttention": FusedQKVAttention,
    },
    # Filter out existing conversions incompatible with new structure
    filtered_weight_conversion_patterns=[".gate_up_proj", ".down_proj"],
    # Add conversions to fuse Q, K, V weights into single qkv_proj
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
    ]
)

# Load model with patches applied
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    patcher=patcher,
)
# Model now has SequentialExperts and FusedQKVAttention modules with properly converted weights
```

## The Patcher class

The [`Patcher`] class handles temporary model patching during [`~PreTrainedModel.from_pretrained`]. It manages both class replacements and weight conversion adjustments needed to load models with modified architectures.

### Parameters

- **class_mapping** (`Dict[str, type[nn.Module]]`) — Mapping from original class names to replacement classes. Class names must exactly match those in the model's module (e.g., `"Qwen2MoeExperts"`, `"LlamaAttention"`).

- **filtered_weight_conversion_patterns** (`str` or `List[str]`, *optional*) — Regex patterns to filter out incompatible weight conversions. Any conversion with source or target patterns matching these will be excluded. Use when structural changes make existing conversions incompatible (e.g., filtering `".gate_up_proj"` when changing expert structure).

- **extra_weight_conversions** (`WeightConverter` or `List[WeightConverter]`, *optional*) — Additional weight conversions to apply for the new structure. These are prepended to existing conversions, allowing you to transform weights to match your modified architecture (e.g., concatenating separate Q, K, V weights into a fused QKV projection).

## Common use cases

### 1. Restructuring MoE experts

Replace complex expert implementations with simpler structures for compatibility with quantization libraries or for easier debugging:

```python
import torch.nn as nn

class SequentialExperts(nn.ModuleList):
    """MoE experts as ModuleList for quantization compatibility."""
    def __init__(self, config):
        super().__init__()
        for _ in range(config.num_experts):
            self.append(ExpertMLP(config))
    
    def forward(self, hidden_states, router_logits, top_k):
        # Sequential processing compatible with quantization libraries
        final_hidden = torch.zeros_like(hidden_states)
        for expert_idx, expert in enumerate(self):
            expert_mask = (router_logits.argmax(dim=-1) == expert_idx)
            if expert_mask.any():
                final_hidden[expert_mask] = expert(hidden_states[expert_mask])
        return final_hidden

patcher = Patcher(
    class_mapping={"Qwen2MoeExperts": SequentialExperts},
    # Filter out conversions for fused expert weights if they existed
    filtered_weight_conversion_patterns=[".gate_up_proj", ".down_proj"],
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    patcher=patcher,
)
```

### 2. Fusing projections for efficiency

Fusing multiple projections into a single layer can enable optimizations like compute-intensive quantization kernels or reduce memory overhead:

```python
class FusedQKVAttention(nn.Module):
    """Fused QKV attention for quantization efficiency."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            3 * config.num_attention_heads * config.head_dim,
            bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False
        )
    
    def forward(self, hidden_states, **kwargs):
        # Split fused QKV after projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        # ... attention computation
        return self.o_proj(output)

patcher = Patcher(
    class_mapping={"Qwen2MoeAttention": FusedQKVAttention},
    # Add fused conversion
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
    ]
)
```

### 3. Experimental modifications

Test alternative implementations without changing source code. This is useful for rapid prototyping and architectural exploration:

```python
class CustomModule(nn.Module):
    """Your experimental implementation of any module."""
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        # Implement your experimental architecture
        # Could be a new attention variant, different MLP structure,
        # alternative normalization, etc.
        self.custom_layer = nn.Linear(config.hidden_size, config.hidden_size)
        ...
    
    def forward(self, *args, **kwargs):
        # Your experimental forward logic
        ...

# Patch any module you want to experiment with
patcher = Patcher(
    class_mapping={
        "ModelBlock": CustomModule,  # Replace any module by name
    }
)

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    patcher=patcher,
)
# Experiment with architectural changes without editing model files
```

## API reference

[[autodoc]] Patcher
