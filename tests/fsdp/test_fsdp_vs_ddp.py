# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""
Tests for FSDP2 vs DDP correctness.
Run:
    pytest tests/fsdp/test_fsdp_vs_ddp.py -v -s
"""

import logging
import os
import socket

import pytest

from transformers import AutoConfig, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import (
    Colors,
    backend_device_count,
    init_test_logger,
    require_fsdp,
    require_torch_multi_accelerator,
    torch_device,
)
from transformers.trainer_utils import set_seed


logger = logging.getLogger("transformers.training_test")


if is_torch_available():
    import tempfile

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.multiprocessing as mp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.tensor import DTensor
    from torch.nn.parallel import DistributedDataParallel as DDP

    from transformers.integrations.fsdp import _find_final_norm, apply_fsdp2, get_transformer_block_classes, initialize_fsdp


# TODO(3outeille): run slow model?
MODEL_NAME = "JackFram/llama-160m"
BATCH_SIZE = 2
SEQ_LEN = 1024
NUM_STEPS = 20
LR = 3e-4
SEED = 42


# =============================================================================
# Distributed helpers
# =============================================================================


def global_wrapper(rank, func, world_size, port, func_args, func_kwargs):
    """Set up distributed environment and run the test function."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # NOTE(3outeille): tells cuBLAS to use a deterministic workspace of 4096 entries x 8 bytes.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def _get_free_port():
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_distributed(world_size: int):
    """Decorator to run function in distributed mode using mp.spawn."""

    def _init_distributed(func):
        def wrapper(*args, **kwargs):
            port = _get_free_port()
            spawn_args = (func, world_size, port, args, kwargs)
            mp.spawn(global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed


def skip_if_insufficient_devices(nproc_per_node):
    """Skip test if not enough GPUs available."""
    if backend_device_count(torch_device) < nproc_per_node:
        pytest.skip(f"Need at least {nproc_per_node} devices, have {backend_device_count(torch_device)}")


# =============================================================================
# Training & comparison helpers
# =============================================================================


def create_deterministic_data(batch_size, seq_len, vocab_size, device, seed):
    """Create deterministic random training data using torch.randint."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = input_ids.clone()
    return [(input_ids, labels)]


def gather_fsdp2_state_dict(model):
    """
    Gather FSDP2 sharded parameters into full tensors via DTensor.full_tensor().
    Uses state_dict() instead of named_parameters() so that tied weights
    (e.g. lm_head.weight == embed_tokens.weight) appear under both names.
    """
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, DTensor):
            state_dict[name] = tensor.full_tensor().clone().detach()
        else:
            state_dict[name] = tensor.clone().detach()
    return state_dict


def compute_grad_norm(model):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            total_norm_sq += grad.data.float().norm(2).item() ** 2
    return total_norm_sq ** 0.5


def log_comparison_table(title, ddp_vals, fsdp_vals):
    """Log a side-by-side comparison table for DDP vs FSDP2 values."""
    C = Colors
    SEP = f"{C.DIM}|{C.RESET}"
    ROW = f"  {C.DIM}{'─' * 52}{C.RESET}"

    logger.info(f"  {C.BOLD}{title}{C.RESET}")
    logger.info(ROW)
    logger.info(
        f"  {C.DIM}{'step':>4}{C.RESET}  "
        f"{SEP}  {C.BLUE}{C.BOLD}{'DDP':^14}{C.RESET}  "
        f"{SEP}  {C.MAGENTA}{C.BOLD}{'FSDP2':^14}{C.RESET}  "
        f"{SEP}  {C.DIM}{'diff':^10}{C.RESET}"
    )
    logger.info(ROW)
    for step in range(len(ddp_vals)):
        diff = abs(ddp_vals[step] - fsdp_vals[step])
        match = f"{C.GREEN}={C.RESET}" if diff < 1e-6 else f"{C.YELLOW}{diff:.1e}{C.RESET}"
        logger.info(
            f"  {C.DIM}{step + 1:>4}{C.RESET}  "
            f"{SEP}  {C.BLUE}{ddp_vals[step]:>14.6f}{C.RESET}  "
            f"{SEP}  {C.MAGENTA}{fsdp_vals[step]:>14.6f}{C.RESET}  "
            f"{SEP}  {match:^10}"
        )
    logger.info(ROW)


def train_ddp(rank, config, batches, lr, device, dtype):
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    ddp_model = DDP(model, device_ids=[rank]).to(dtype)
    ddp_model.train()

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = ddp_model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        grad_norm = compute_grad_norm(ddp_model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = {k: v.clone().detach() for k, v in ddp_model.module.state_dict().items()}
    return losses, grad_norms, state_dict


def train_fsdp2(rank, config, batches, lr, device_map, device_mesh, dtype, fsdp_plan):
    """Run an FSDP2 training loop with Adam.

    Returns (losses, grad_norms, state_dict).
    """
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
    model = apply_fsdp2(model, device_mesh, fsdp_plan=fsdp_plan)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = gather_fsdp2_state_dict(model)
    return losses, grad_norms, state_dict


def assert_ddp_fsdp_match(ddp_losses, ddp_grad_norms, ddp_state_dict, fsdp_losses, fsdp_grad_norms, fsdp_state_dict, label="FSDP2"):
    """Assert that DDP and FSDP2 training produced identical results."""
    for step in range(len(ddp_losses)):
        torch.testing.assert_close(
            torch.tensor(ddp_losses[step]),
            torch.tensor(fsdp_losses[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Step {step} loss mismatch: DDP={ddp_losses[step]}, {label}={fsdp_losses[step]}",
        )

    for step in range(len(ddp_grad_norms)):
        torch.testing.assert_close(
            torch.tensor(ddp_grad_norms[step]),
            torch.tensor(fsdp_grad_norms[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Step {step} grad norm mismatch: DDP={ddp_grad_norms[step]}, {label}={fsdp_grad_norms[step]}",
        )

    for key in ddp_state_dict:
        assert key in fsdp_state_dict, f"Key {key} missing from {label} state dict"
        torch.testing.assert_close(
            ddp_state_dict[key],
            fsdp_state_dict[key],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Weight mismatch for {key}",
        )


# =============================================================================
# Test: FSDP2 sharding structure (TorchTitan parity)
# =============================================================================


def _test_fsdp2_sharding_structure_impl(rank, tie_word_embeddings):
    """
    Verify that apply_fsdp2(fsdp_plan="auto") wraps exactly the right modules.

    Expected FSDP targets:
    UNTIED                              TIED
    ──────                              ────
    1. embed_tokens  (reshard=True)     1. (skip — embed goes to step 3)
    2. layers[i]     (reshard=True)     2. layers[i]     (reshard=True)
    3. [norm, lm_head] (reshard=False)  3. [norm, embed_tokens] (reshard=False)
    4. root                             4. root

    Detection: fully_shard(module) swaps module.__class__ to an FSDP-prefixed
    class (e.g. LlamaDecoderLayer → FSDPLlamaDecoderLayer). We collect all
    modules whose class name starts with "FSDP" and compare against the
    expected set.
    """
    init_test_logger()

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.tie_word_embeddings = tie_word_embeddings

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")

    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)

    # --- Before FSDP: build the expected set of FSDP target names ---
    block_classes = get_transformer_block_classes(model)
    assert block_classes, "get_transformer_block_classes found no block classes"

    decoder_layer_names = {
        name for name, module in model.named_modules() if type(module) in block_classes
    }
    assert len(decoder_layer_names) == config.num_hidden_layers

    # Resolve module names via id() since get_input_embeddings() etc. return
    # the module object, not its name in the module tree.
    id_to_name = {id(module): name for name, module in model.named_modules()}

    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    final_norm = _find_final_norm(model, decoder_layer_names)
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )

    embed_name = id_to_name.get(id(input_embed))
    output_name = id_to_name.get(id(output_embed))
    norm_name = id_to_name.get(id(final_norm))

    # Build expected set using set union (|) operator:
    # {"a"} | {"b"} == {"a", "b"}
    expected_targets = (
        {""}                     # root
        | decoder_layer_names    # each decoder layer
        | {embed_name}           # embeddings (always — own bucket or grouped with norm)
        | {norm_name}            # final norm
    )
    if not weights_tied:
        # Untied: lm_head gets its own FSDP wrap (grouped with norm)
        expected_targets |= {output_name}
    # Tied: lm_head.weight IS embed_tokens.weight → already in [embed, norm] bucket

    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")

    actual_targets = {
        name for name, module in model.named_modules()
        if type(module).__name__.startswith("FSDP")
    }

    if rank == 0:
        logger.info(f"  Weights tied: {weights_tied}")
        logger.info(f"  Expected FSDP targets: {sorted(expected_targets)}")
        logger.info(f"  Actual FSDP targets:   {sorted(actual_targets)}")

    missing = expected_targets - actual_targets
    extra = actual_targets - expected_targets
    assert not missing and not extra, (
        f"FSDP target mismatch.\n"
        f"  Missing (expected but not wrapped): {sorted(missing)}\n"
        f"  Extra (wrapped but not expected):   {sorted(extra)}"
    )

    if rank == 0:
        logger.info(f"  FSDP sharding structure OK ({len(actual_targets)} targets)")


@pytest.mark.parametrize("nproc_per_node", [pytest.param(2, id="2gpus")])
@pytest.mark.parametrize(
    "tie_word_embeddings",
    [pytest.param(False, id="untied"), pytest.param(True, id="tied")],
)
@require_fsdp
@require_torch_multi_accelerator
def test_fsdp2_sharding_structure(nproc_per_node, tie_word_embeddings):
    """Verify per-decoder-layer FSDP sharding matches TorchTitan's granularity."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(world_size=nproc_per_node)(_test_fsdp2_sharding_structure_impl)(tie_word_embeddings)


# =============================================================================
# Test: FSDP2 auto plan vs DDP
# =============================================================================


def _test_fsdp2_auto_plan_vs_ddp_impl(rank, dtype, tie_word_embeddings):
    """Compare losses, grad norms, and final weights between DDP and FSDP2."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.tie_word_embeddings = tie_word_embeddings

    batches = create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = train_ddp(rank, config, batches, LR, device, dtype)

    dist.barrier()

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
        rank, config, batches, LR, device_map, device_mesh, dtype, fsdp_plan="auto",
    )

    dist.barrier()

    if rank == 0:
        logger.info("")
        log_comparison_table("Loss per step", ddp_losses, fsdp_losses)
        logger.info("")
        log_comparison_table("Gradient norm per step", ddp_grad_norms, fsdp_grad_norms)
        logger.info("")

    assert_ddp_fsdp_match(ddp_losses, ddp_grad_norms, ddp_state_dict, fsdp_losses, fsdp_grad_norms, fsdp_state_dict)


@pytest.mark.parametrize("nproc_per_node", [pytest.param(2, id="2gpus")])
@pytest.mark.parametrize(
    "tie_word_embeddings",
    [pytest.param(False, id="untied"), pytest.param(True, id="tied")],
)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float32, id="float32"), pytest.param(torch.bfloat16, id="bfloat16")],
)
@require_fsdp
@require_torch_multi_accelerator
def test_fsdp2_auto_plan_vs_ddp(nproc_per_node, tie_word_embeddings, dtype):
    """20-step Adam: compare per-step losses, grad norms, and final weights."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(world_size=nproc_per_node)(_test_fsdp2_auto_plan_vs_ddp_impl)(dtype, tie_word_embeddings)


# =============================================================================
# Test: FSDP2 manual plan vs DDP
# =============================================================================


def _test_fsdp2_manual_plan_vs_ddp_impl(rank, dtype, tie_word_embeddings):
    """Compare DDP vs FSDP2 with a per-sublayer manual plan (self_attn + mlp buckets)."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.tie_word_embeddings = tie_word_embeddings

    batches = create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = train_ddp(rank, config, batches, LR, device, dtype)

    dist.barrier()

    # Fine-grained plan: shard at self_attn and mlp level instead of whole layers.
    fsdp_plan = {"model.embed_tokens": "free_full_weight"}
    for i in range(config.num_hidden_layers):
        fsdp_plan[f"model.layers.{i}.self_attn"] = "free_full_weight"
        fsdp_plan[f"model.layers.{i}.mlp"] = "free_full_weight"
    fsdp_plan["model.norm"] = "keep_full_weight"
    if not tie_word_embeddings:
        fsdp_plan["lm_head"] = "keep_full_weight"
    
    print(fsdp_plan)

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
        rank, config, batches, LR, device_map, device_mesh, dtype, fsdp_plan=fsdp_plan,
    )

    dist.barrier()

    if rank == 0:
        logger.info("")
        log_comparison_table("Loss per step (manual plan)", ddp_losses, fsdp_losses)
        logger.info("")
        log_comparison_table("Gradient norm per step (manual plan)", ddp_grad_norms, fsdp_grad_norms)
        logger.info("")

    assert_ddp_fsdp_match(
        ddp_losses, ddp_grad_norms, ddp_state_dict,
        fsdp_losses, fsdp_grad_norms, fsdp_state_dict,
        label="FSDP2(manual)",
    )


@pytest.mark.parametrize("nproc_per_node", [pytest.param(2, id="2gpus")])
@pytest.mark.parametrize(
    "tie_word_embeddings",
    [pytest.param(False, id="untied"), pytest.param(True, id="tied")],
)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float32, id="float32"), pytest.param(torch.bfloat16, id="bfloat16")],
)
@require_fsdp
@require_torch_multi_accelerator
def test_fsdp2_manual_plan_vs_ddp(nproc_per_node, tie_word_embeddings, dtype):
    """20-step Adam with manual fsdp_plan dict: compare per-step losses, grad norms, and final weights."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(world_size=nproc_per_node)(_test_fsdp2_manual_plan_vs_ddp_impl)(dtype, tie_word_embeddings)


# =============================================================================
# Test: FSDP2 save/load checkpoint
# =============================================================================


def _test_fsdp2_save_load_impl(rank):
    """Train FSDP2 model, save via DCP, load into fresh model, compare state dicts."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Train for a few steps so state is non-trivial
    batches = create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")

    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)
    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()

    # Gather full state dict before saving
    state_dict_before = gather_fsdp2_state_dict(model)

    # Save checkpoint via DCP
    # Use a shared tmpdir across all ranks: create on rank 0, broadcast to others
    if rank == 0:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        tmpdir_list = [tmpdir]
    else:
        tmpdir_list = [None]
    dist.broadcast_object_list(tmpdir_list, src=0)
    tmpdir = tmpdir_list[0]

    try:
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        dcp.save({"model": model_state_dict, "optim": optimizer_state_dict}, checkpoint_id=tmpdir)
        dist.barrier()

        # Create a fresh FSDP2 model + optimizer and load the checkpoint
        set_seed(SEED)
        new_model = AutoModelForCausalLM.from_config(config).to(device_map)
        new_model = apply_fsdp2(new_model, device_mesh, fsdp_plan="auto")
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=LR)

        new_model_state_dict, new_optimizer_state_dict = get_state_dict(new_model, new_optimizer)
        dcp.load({"model": new_model_state_dict, "optim": new_optimizer_state_dict}, checkpoint_id=tmpdir)
        set_state_dict(new_model, new_optimizer, model_state_dict=new_model_state_dict, optim_state_dict=new_optimizer_state_dict)
        dist.barrier()
    finally:
        if rank == 0:
            tmpdir_obj.cleanup()

    # Gather full state dict after loading
    state_dict_after = gather_fsdp2_state_dict(new_model)

    # Compare model weights
    for key in state_dict_before:
        assert key in state_dict_after, f"Key {key} missing after load"
        torch.testing.assert_close(
            state_dict_before[key],
            state_dict_after[key],
            rtol=0,
            atol=0,
            msg=f"Weight mismatch for {key} after save/load",
        )

    if rank == 0:
        logger.info(f"FSDP2 save/load test passed: all {len(state_dict_before)} parameters match exactly.")


@pytest.mark.parametrize("nproc_per_node", [pytest.param(2, id="2gpus")])
@require_fsdp
@require_torch_multi_accelerator
def test_fsdp2_save_load(nproc_per_node):
    """Save FSDP2 checkpoint via DCP, load into fresh model, verify exact match."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(world_size=nproc_per_node)(_test_fsdp2_save_load_impl)()
