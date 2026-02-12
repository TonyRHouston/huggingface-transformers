# Dynamic Weight Loading in Transformers

This document provides a comprehensive explanation of the dynamic weight loading system in the Hugging Face Transformers library. This system enables efficient loading of model checkpoints with on-the-fly weight transformations, tensor parallelism support, and quantization integration.

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Architecture](#architecture)
3. [WeightTransform, WeightRenaming & WeightConverter](#weighttransform-weightrenaming--weightconverter)
4. [ConversionOps](#conversionops)
5. [Operation Chaining](#operation-chaining)
6. [Tensor Parallelism Integration](#tensor-parallelism-integration)
7. [Quantization Integration](#quantization-integration)
8. [Async Loading & Scheduling](#async-loading--scheduling)
9. [Reversibility](#reversibility)
10. [Real Examples](#real-examples)

---

## Overview & Motivation

### Why Dynamic Weight Loading?

Modern transformer models often have checkpoint formats that differ from their runtime representations. Common scenarios include:

1. **Fused Weights**: Checkpoints store separate `gate_proj` and `up_proj` weights, but the model uses a fused `gate_up_proj` for efficiency
2. **MoE Expert Consolidation**: Individual expert weights (`experts.0.weight`, `experts.1.weight`, ...) need to be stacked into a single 3D tensor
3. **Legacy Naming**: Old checkpoints use different naming conventions (e.g., `LayerNorm.gamma` vs `LayerNorm.weight`)
4. **Quantization**: Weights may be stored in quantized formats that need deserialization

The dynamic weight loading system solves these problems by:
- Transforming weights **during** loading (not after)
- Supporting asynchronous I/O for better performance
- Integrating seamlessly with tensor parallelism
- Enabling round-trip save/load through reversible operations

---

## Full Pipeline: Dense vs MoE Models

### Key Distinction

It's important to understand the difference between:

1. **Dynamic weight loading** (used by ALL models) - the general loading pipeline
2. **Conversion mapping** (used by SOME models) - weight format transformations

All models go through the dynamic weight loading system. Conversion mapping is an **optional step within that system** that only activates when the model has entries in `_MODEL_TO_CONVERSION_PATTERN`.

### Full Weight Loading Pipeline

```
Checkpoint File → from_pretrained() → convert_and_load_state_dict_in_model()
                                              ↓
                         ┌──────────────────────────────────────┐
                         │  For each weight in checkpoint:      │
                         │  1. Match key to model parameter     │
                         │  2. Apply conversion (if defined)    │
                         │  3. Apply TP sharding (if tp_plan)   │
                         │  4. Apply quantization (if enabled)  │
                         │  5. Set parameter on model           │
                         └──────────────────────────────────────┘
```

### Dense Model Example (e.g., Llama)

**Checkpoint format == Model format** (no conversion needed)

```
Checkpoint:                    Model:
q_proj.weight        →        q_proj.weight
k_proj.weight        →        k_proj.weight
v_proj.weight        →        v_proj.weight
gate_proj.weight     →        gate_proj.weight
up_proj.weight       →        up_proj.weight
```

- **No conversion mapping needed** - keys match directly
- **TP sharding still applies** - weights are sharded based on `tp_plan`

### MoE Model Example (e.g., Mixtral)

**Checkpoint format ≠ Model format** (conversion required)

```
Checkpoint:                              Model:
experts.0.w1.weight  ─┐
experts.1.w1.weight   │ MergeModulelist
...                   ├───────────────→  experts.gate_up_proj (8, hidden, 2*intermediate)
experts.0.w3.weight   │ + Concatenate
experts.1.w3.weight  ─┘
```

- **Conversion mapping needed** - transforms separate expert weights into fused 3D tensors
- **TP sharding applies after conversion** - shards the fused tensor

### Pipeline Comparison Table

| Model Type | Dynamic Loading | Conversion Mapping | TP Sharding |
|------------|-----------------|-------------------|-------------|
| Dense (Llama, Mistral) | ✅ | ❌ (not needed) | ✅ |
| MoE (Mixtral, Qwen2-MoE) | ✅ | ✅ (fuses experts) | ✅ |

### When Each Step Activates

1. **Dynamic loading**: Always active for all models
2. **Conversion mapping**: Only when `model_type` is in `_MODEL_TO_CONVERSION_PATTERN`
3. **TP sharding**: Only when `tp_plan="auto"` and model has `base_model_tp_plan`
4. **Quantization**: Only when quantization config is provided

---

## Architecture

### Core Components

The system is built around several key components defined in `src/transformers/core_model_loading.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                     convert_and_load_state_dict_in_model        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ WeightRenaming│    │WeightConverter│    │ ConversionOps   │  │
│  │              │    │              │    │                  │  │
│  │ Simple key   │    │ Multi-step   │    │ - Chunk          │  │
│  │ renaming     │    │ transforms   │    │ - Concatenate    │  │
│  │              │    │              │    │ - MergeModulelist│  │
│  └──────────────┘    └──────────────┘    │ - Transpose      │  │
│                                          │ - etc.           │  │
│                                          └──────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ThreadPoolExecutor                       │  │
│  │           (Async tensor materialization)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

**`WeightTransform`** (base dataclass):
```python
@dataclass(slots=True)
class WeightTransform:
    source_patterns: str | list[str]      # Checkpoint key patterns
    target_patterns: str | list[str]      # Model key patterns
    compiled_sources: re.Pattern          # Compiled regex for matching
    distributed_operation: TensorParallelLayer | None
    quantization_operation: ConversionOps | None
    collected_tensors: dict[str, list[Future]]  # Gathered tensors
    layer_targets: dict[str, set[str]]          # Target key tracking
```

---

## WeightTransform, WeightRenaming & WeightConverter

### WeightTransform

The base class that handles pattern matching and tensor collection. It provides:

- **Pattern compilation**: Converts glob-style patterns (`*.weight`) to regex
- **Key renaming**: `rename_source_key()` transforms checkpoint keys to model keys
- **Tensor collection**: `add_tensor()` gathers related tensors for batch processing
- **Reversibility**: `reverse_transform()` creates the inverse operation for saving

### WeightRenaming

A specialized `WeightTransform` for simple key renaming without tensor operations:

```python
@dataclass(slots=True)
class WeightRenaming(WeightTransform):
    # Simple 1:1 key renaming
    # Example: "LayerNorm.gamma" -> "LayerNorm.weight"
```

Use cases:
- Legacy checkpoint compatibility (`LayerNorm.gamma` -> `LayerNorm.weight`)
- Module path changes (`.block_sparse_moe.` -> `.mlp.`)
- Adding prefixes (`(.+)` -> `timm_model.\1`)

### WeightConverter

Extends `WeightTransform` with a list of `ConversionOps`:

```python
@dataclass(slots=True)
class WeightConverter(WeightTransform):
    operations: list[ConversionOps]  # Chain of operations
```

Key features:
- Supports many-to-one (e.g., concatenating `gate` + `up` -> `gate_up`)
- Supports one-to-many (e.g., splitting `qkv` -> `q`, `k`, `v`)
- Operations are applied sequentially

---

## ConversionOps

### Base Class

```python
class ConversionOps:
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs) -> dict:
        """Transform tensors according to the operation."""
        raise NotImplementedError

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse operation for saving."""
        raise NotImplementedError
```

### Available Operations

#### Chunk
Splits a tensor into equal parts along a dimension:

```python
class Chunk(ConversionOps):
    def __init__(self, dim: int = 0):
        self.dim = dim
```

**Use case**: Split fused `qkv` into separate `q`, `k`, `v` tensors

**Reverse**: `Concatenate`

#### Concatenate
Joins multiple tensors along a dimension:

```python
class Concatenate(ConversionOps):
    def __init__(self, dim: int = 0):
        self.dim = dim
```

**Use case**: Fuse `gate_proj` and `up_proj` into `gate_up_proj`

**Reverse**: `Chunk`

#### MergeModulelist
Stacks a list of 2D tensors into a single 3D tensor:

```python
class MergeModulelist(ConversionOps):
    def __init__(self, dim: int = 0):
        self.dim = dim
```

**Use case**: Stack individual expert weights `[expert_0, expert_1, ...]` into `(num_experts, in_features, out_features)`

**Reverse**: `SplitModulelist`

#### SplitModulelist
Unstacks a 3D tensor back into a list of 2D tensors:

```python
class SplitModulelist(ConversionOps):
    def __init__(self, dim: int = 0):
        self.dim = dim
```

**Use case**: Save stacked expert weights as individual tensors

**Reverse**: `MergeModulelist`

#### Transpose
Swaps dimensions of a tensor:

```python
class Transpose(ConversionOps):
    def __init__(self, dim0: int = 0, dim1: int = 1):
        self.dim0 = dim0
        self.dim1 = dim1
```

**Use case**: Convert weight layouts between different conventions

**Reverse**: `Transpose(dim1, dim0)`

#### PermuteForRope
Applies permutation for RoPE (Rotary Position Embedding) weight conversion:

```python
class PermuteForRope(ConversionOps):
    # Converts complex RoPE weights to split sin/cos format
```

#### Force16BytesAlignment
Ensures tensor memory alignment for optimized kernels:

```python
class Force16BytesAlignment(ConversionOps):
    # Clones tensor if not 16-byte aligned
    # Required for torch._grouped_mm and TMA/SIMD operations
```

**Reverse**: `Force16BytesAlignment` (idempotent)

#### ErnieFuseAndSplitTextVisionExperts
Specialized operation for ERNIE 4.5 VL MoE models:

```python
class ErnieFuseAndSplitTextVisionExperts(ConversionOps):
    # Splits experts over keys and fuses over modules
    # For handling text/vision expert separation
```

---

## Operation Chaining

Operations can be chained to perform complex transformations. The operations execute in order, with each operation's output becoming the next operation's input.

### Example: Mixtral MoE Conversion

```python
WeightConverter(
    source_patterns=[
        ".experts.*.w1.weight",  # gate_proj per expert
        ".experts.*.w3.weight",  # up_proj per expert
    ],
    target_patterns=".experts.gate_up_proj",
    operations=[
        MergeModulelist(dim=0),  # Stack all experts: (n_experts, in, out)
        Concatenate(dim=1),      # Fuse gate+up: (n_experts, in, 2*out)
    ],
)
```

**Data flow**:
```
Input:
  ".experts.*.w1.weight": [tensor_0, tensor_1, ..., tensor_7]  # 8 experts
  ".experts.*.w3.weight": [tensor_0, tensor_1, ..., tensor_7]  # 8 experts

After MergeModulelist(dim=0):
  ".experts.*.w1.weight": (8, 4096, 14336)  # stacked gate
  ".experts.*.w3.weight": (8, 4096, 14336)  # stacked up

After Concatenate(dim=1):
  ".experts.gate_up_proj": (8, 4096, 28672)  # fused gate_up
```

### Pattern Matching Details

The `*` in patterns acts as a wildcard:
- During loading: matches any numeric index (`experts.0.`, `experts.1.`, etc.)
- Tensors with the same pattern (differing only in index) are grouped together
- The order of collection is preserved for correct concatenation

---

## Tensor Parallelism Integration

### Overview

The dynamic loading system integrates with tensor parallelism (TP) through the `TensorParallelLayer` hierarchy defined in `src/transformers/integrations/tensor_parallel.py`.

### Sharding During Load

When TP is enabled, tensors are sharded **during** materialization, not after:

```python
def spawn_tp_materialize(thread_pool, tensor, sharding_method, tensor_idx, device, dtype):
    def _job():
        return sharding_method.shard_tensor(tensor, tensor_idx=tensor_idx, device=device, dtype=dtype)
    return thread_pool.submit(_job)
```

This means each rank only loads the portion of the tensor it needs.

### Available Parallel Styles

| Style | Weight Shard Dim | Description |
|-------|------------------|-------------|
| `colwise` | -2 | Column-wise: output features sharded |
| `rowwise` | -1 | Row-wise: input features sharded |
| `packed_colwise` | -2 | For fused weights (gate_up_proj) |
| `packed_rowwise` | -1 | For fused weights |
| `embedding_rowwise` | 0 | Vocabulary parallelism |
| `grouped_gemm` | 0 | Expert parallelism for MoE |
| `sequence_parallel` | None | No weight sharding |

### Packed Weight Handling

For fused weights like `gate_up_proj`, special care is needed to shard correctly:

```python
def get_packed_weights(param, empty_param, device_mesh, rank, dim):
    """
    Interleaves gate and up shards correctly.

    Packed tensor: [G0 G1 G2 G3 | U0 U1 U2 U3]

    With TP=2:
    - Rank 0 gets: [G0 G1 | U0 U1]
    - Rank 1 gets: [G2 G3 | U2 U3]
    """
```

### Integration with WeightConverter

The TP operation is stored in the `WeightTransform`:

```python
if matched_tp_pattern := tp_plan_alt.search(renamed_key):
    tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]]
    mapping.distributed_operation = tp_layer(
        device_mesh=device_mesh,
        rank=device_mesh.get_local_rank(),
        empty_param=empty_param.clone()
    )
```

---

## Quantization Integration

### Overview

Quantization is integrated through the `HfQuantizer` class in `src/transformers/quantizers/base.py`. Quantizers can provide:

1. **Quantization operations** for on-the-fly quantization during load
2. **Weight conversions** for deserializing pre-quantized checkpoints

### Pre-quantized Loading

For pre-quantized models, the quantizer provides `WeightConverter` instances:

```python
def get_weight_conversions(self):
    """Returns list of WeightConverter for deserializing quantized weights."""
    return []  # Override in subclass
```

Example for TorchAO:
```python
WeightConverter(
    source_patterns=[":qdata", ":scale"],
    target_patterns="",
    operations=[TorchaoDeserialize()],
)
```

### On-the-fly Quantization

For non-pre-quantized models, the quantizer provides a quantization operation:

```python
def get_quantize_ops(self):
    """Returns ConversionOps for quantizing weights."""
    raise NotImplementedError
```

This is applied after other conversions:

```python
if hf_quantizer is not None and mapping.quantization_operation is not None:
    collected_tensors = mapping.quantization_operation.convert(
        collected_tensors,
        source_patterns=...,
        target_patterns=...,
        model=model,
        config=config,
    )
```

### Dtype Handling

The system preserves checkpoint dtypes for pre-quantized weights:

```python
if hf_quantizer and hf_quantizer.pre_quantized and original_key != renamed_key:
    # Key was renamed during deserialization, preserve original dtype
    _dtype = None
```

---

## Async Loading & Scheduling

### Thread Pool Configuration

```python
GLOBAL_WORKERS = min(4, os.cpu_count() or 4)
```

The system uses a limited thread pool (default 4 workers) because:
- I/O bound operations benefit from some parallelism
- Too many threads (e.g., 16) can **double** loading time
- Memory must be managed carefully

### Async vs Sync Loading

```python
def spawn_materialize(thread_pool, tensor, device, dtype) -> Future | Callable:
    def _job():
        return _materialize_copy(tensor, device, dtype)

    if thread_pool is not None:
        return thread_pool.submit(_job)  # Async: returns Future
    else:
        return _job  # Sync: returns Callable (deferred execution)
```

Sync loading is used when:
- `HF_DEACTIVATE_ASYNC_LOAD=1` environment variable is set
- Disk offloading is enabled (memory constraints require sequential loading)

### Materialization Flow

```
1. Checkpoint iteration:
   - For each key, submit materialization job
   - Job returns Future (async) or Callable (sync)
   - Add to WeightConverter.collected_tensors

2. Conversion phase:
   - materialize_tensors() waits for all Futures
   - Applies conversion operations
   - Sets parameters on model

3. Cleanup:
   - Delete realized tensors immediately
   - Thread pool shutdown (with cancel_futures=True for interrupts)
```

### Memory Efficiency

The system minimizes memory usage through:

1. **Deferred loading**: Tensors aren't loaded until needed
2. **Immediate cleanup**: `del realized_value` after setting parameters
3. **Sequential fallback**: For disk offloading, loads one tensor at a time

---

## Reversibility

### Save/Load Round-Trip

The system supports saving models with the inverse transformations:

```python
def revert_weight_conversion(model, state_dict):
    """Applies reverse conversions for saving."""
    weight_conversions = getattr(model, "_weight_conversions", None)

    # Reverse all transforms
    reverse_weight_conversion = [
        conversion.reverse_transform() for conversion in weight_conversions
    ]

    # Apply in reverse
    for first_param_name, reversed_converter in conversion_mapping.items():
        realized_value = reversed_converter.convert(first_param_name, model=model)
```

### How Reversibility Works

Each `ConversionOps` defines its inverse:

| Operation | Reverse |
|-----------|---------|
| `Chunk(dim)` | `Concatenate(dim)` |
| `Concatenate(dim)` | `Chunk(dim)` |
| `MergeModulelist(dim)` | `SplitModulelist(dim)` |
| `SplitModulelist(dim)` | `MergeModulelist(dim)` |
| `Transpose(d0, d1)` | `Transpose(d1, d0)` |

### Pattern Processing for Reverse

Target patterns may contain regex elements that need processing:

```python
def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    - Removes `^` and `$` anchors
    - Removes negative lookahead/lookbehind
    - Detects capturing groups, replaces with \1
    """
```

---

## Real Examples

### Mixtral-style MoE

**Checkpoint format**:
```
model.layers.0.block_sparse_moe.experts.0.w1.weight  # gate per expert
model.layers.0.block_sparse_moe.experts.0.w2.weight  # down per expert
model.layers.0.block_sparse_moe.experts.0.w3.weight  # up per expert
...
model.layers.0.block_sparse_moe.experts.7.w1.weight
```

**Model format**:
```
model.layers.0.mlp.experts.gate_up_proj  # (8, 4096, 28672)
model.layers.0.mlp.experts.down_proj     # (8, 14336, 4096)
```

**Conversion mapping** (from `conversion_mapping.py`):
```python
"mixtral": [
    WeightRenaming(".block_sparse_moe.", ".mlp."),
    WeightConverter(
        source_patterns=[".experts.*.w1.weight", ".experts.*.w3.weight"],
        target_patterns=".experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    ),
    WeightConverter(
        source_patterns=[".experts.*.w2.weight"],
        target_patterns=".experts.down_proj",
        operations=[MergeModulelist(dim=0)],
    ),
],
```

### Qwen2-style MoE

**Checkpoint format**:
```
model.layers.0.mlp.experts.0.gate_proj.weight
model.layers.0.mlp.experts.0.up_proj.weight
model.layers.0.mlp.experts.0.down_proj.weight
...
```

**Model format**: Same as Mixtral

**Conversion mapping**:
```python
"qwen2_moe": [
    WeightConverter(
        source_patterns=[
            "mlp.experts.*.gate_proj.weight",
            "mlp.experts.*.up_proj.weight",
        ],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    ),
    WeightConverter(
        source_patterns="mlp.experts.*.down_proj.weight",
        target_patterns="mlp.experts.down_proj",
        operations=[MergeModulelist(dim=0)],
    ),
],
```

### Model Type Aliases

Many models share conversion patterns:

```python
_MODEL_TO_CONVERSION_PATTERN = {
    "mixtral": "mixtral",
    "minimax": "mixtral",
    "qwen2_moe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "qwen3_moe": "qwen2_moe",
    "olmoe": "qwen2_moe",
    ...
}
```

### ERNIE 4.5 VL MoE (Complex Example)

This model has text and vision experts that need special handling:

```python
"ernie4_5_vl_moe": [
    # Vision model renaming
    WeightRenaming("vision_model", "vision_tower"),

    # Gate weight transposition
    WeightConverter(
        source_patterns="mlp.gate.weight",
        target_patterns="mlp.text_moe.gate.weight",
        operations=[Transpose(dim0=0, dim1=1)],
    ),

    # Split experts between text and vision
    WeightConverter(
        source_patterns=["experts.*.down_proj.weight"],
        target_patterns=[
            "text_moe.experts.down_proj",
            "vision_moe.experts.down_proj",
        ],
        operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
    ),
],
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/transformers/core_model_loading.py` | Core loading logic, WeightConverter, ConversionOps |
| `src/transformers/conversion_mapping.py` | Built-in conversion patterns for all models |
| `src/transformers/integrations/tensor_parallel.py` | TP sharding classes and utilities |
| `src/transformers/quantizers/base.py` | Quantization hooks and base class |

---

## Summary

The dynamic weight loading system provides:

1. **Flexibility**: Handle any checkpoint format through composable operations
2. **Performance**: Async I/O and on-the-fly sharding minimize memory and time
3. **Correctness**: Reversible operations ensure save/load round-trips work
4. **Integration**: Seamless support for TP, EP, and quantization

This architecture enables Transformers to support a wide variety of model formats while maintaining a clean, efficient loading path.
