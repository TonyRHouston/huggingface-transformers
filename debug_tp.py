"""Quick debug script to understand the TP crash."""
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size, model_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    from transformers import GptOssForCausalLM
    from transformers import set_seed

    set_seed(0)

    # Enable logging to see TP plan resolution
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Load TP model
    model_tp = GptOssForCausalLM.from_pretrained(model_path, tp_plan="auto")
    device = model_tp.device

    # Print shapes of ALL parameters in layer 0
    for name, param in model_tp.named_parameters():
        if "layers.0" in name:
            print(f"[Rank {rank}] {name}: {param.shape}", flush=True)

    # Print num_experts
    experts = model_tp.model.layers[0].mlp.experts
    print(f"[Rank {rank}] num_experts: {experts.num_experts}")

    model_tp.train()
    set_seed(42)
    vocab_size = model_tp.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    set_seed(43)
    labels = torch.randint(0, vocab_size, (2, 64)).to(device)

    try:
        loss = model_tp(input_ids, labels=labels).loss
        print(f"[Rank {rank}] Forward passed! Loss: {loss.item()}")
        loss.backward()
        print(f"[Rank {rank}] Backward passed!")
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()

    dist.destroy_process_group()

if __name__ == "__main__":
    from transformers import GptOssForCausalLM, GptOssConfig

    # Create and save model
    config = GptOssConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=99,
        max_position_embeddings=512,
        pad_token_id=0,
    )
    print(f"Config num_local_experts: {config.num_local_experts}")
    model = GptOssForCausalLM(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        mp.spawn(run, args=(2, tmp_dir), nprocs=2, join=True)
