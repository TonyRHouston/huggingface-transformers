from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from torch.distributed.elastic.multiprocessing.errors import record

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
# model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")
# Need to be initialized explicitly to use the `barrier` before loading
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=rank)

@record
def main():

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan="auto")
    # model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {"role": "user", "content": "What do you think about life?"},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    input_size = inputs.input_ids.shape[-1]
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    text = tokenizer.batch_decode(output[:, input_size:])[0]
    print(text)

main()

torch.distributed.destroy_process_group()