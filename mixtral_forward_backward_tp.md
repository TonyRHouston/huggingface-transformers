# Mixtral Single Layer Forward/Backward with TP=2

## Setup

```
hidden_size = 4
intermediate_size = 4
num_experts = 2, top_k = 1
TP = 2 (GPU0, GPU1)
1 token input
```

## Fake Data

```python
# Input (replicated on both GPUs)
x = [1.0, 2.0, 3.0, 4.0]  # shape [1, 4]

# Router routes token to expert 0 with weight 0.6
routing_weight = 0.6
expert_idx = 0
```

## Weight Sharding

```
gate_up_proj original: [2, 8, 4]  (2 experts, 2*intermediate, hidden)
  Expert 0: [[g0, g1, g2, g3],    ← gate rows
             [g4, g5, g6, g7],
             [u0, u1, u2, u3],    ← up rows
             [u4, u5, u6, u7]]

packed_colwise splits on dim -2 (interleaved):
  GPU0: [2, 4, 4] → [[g0,g1], [g4,g5], [u0,u1], [u4,u5]]  (first half of gate + first half of up)
  GPU1: [2, 4, 4] → [[g2,g3], [g6,g7], [u2,u3], [u6,u7]]  (second half of gate + second half of up)

down_proj original: [2, 4, 4]  (2 experts, hidden, intermediate)
rowwise splits on dim -1:
  GPU0: [2, 4, 2]  (first half of intermediate)
  GPU1: [2, 4, 2]  (second half of intermediate)
```

## Concrete Weights

```python
# GPU0                                    # GPU1
gate_up_0 = [[0.1, 0.2, 0.3, 0.4],       gate_up_1 = [[0.5, 0.6, 0.7, 0.8],
             [0.1, 0.2, 0.3, 0.4],                    [0.5, 0.6, 0.7, 0.8],
             [0.1, 0.1, 0.1, 0.1],                    [0.2, 0.2, 0.2, 0.2],
             [0.1, 0.1, 0.1, 0.1]]                    [0.2, 0.2, 0.2, 0.2]]

down_0 = [[0.1, 0.1],                    down_1 = [[0.1, 0.1],
          [0.1, 0.1],                              [0.1, 0.1],
          [0.1, 0.1],                              [0.1, 0.1],
          [0.1, 0.1]]                              [0.1, 0.1]]
```

---

## FORWARD PASS

### F1: moe_tp_experts input hook
```
all_reduce_backward(x) → identity in forward → x unchanged
x = [1, 2, 3, 4] on both GPUs
```

### F2: gate_up projection
```
GPU0: gate_up_out = x @ gate_up_0.T      GPU1: gate_up_out = x @ gate_up_1.T
      = [1,2,3,4] @ [[.1,.2,.3,.4],            = [1,2,3,4] @ [[.5,.6,.7,.8],
                     [.1,.2,.3,.4],                          [.5,.6,.7,.8],
                     [.1,.1,.1,.1],                          [.2,.2,.2,.2],
                     [.1,.1,.1,.1]].T                        [.2,.2,.2,.2]].T
      = [3.0, 3.0, 1.0, 1.0]                   = [7.0, 7.0, 2.0, 2.0]
        [g0, g1,  u0,  u1]                      [g2, g3,  u2,  u3]
```

### F3: split into gate and up, apply silu(gate) * up
```
GPU0: gate=[3.0, 3.0], up=[1.0, 1.0]     GPU1: gate=[7.0, 7.0], up=[2.0, 2.0]
      silu(gate) ≈ [2.86, 2.86]                silu(gate) ≈ [6.99, 6.99]
      inter = [2.86, 2.86]                     inter = [13.98, 13.98]
```

### F4: down projection
```
GPU0: partial = inter @ down_0.T         GPU1: partial = inter @ down_1.T
      = [2.86, 2.86] @ [[.1,.1],               = [13.98, 13.98] @ [[.1,.1],
                        [.1,.1],                                   [.1,.1],
                        [.1,.1],                                   [.1,.1],
                        [.1,.1]].T                                 [.1,.1]].T
      = [0.57, 0.57, 0.57, 0.57]               = [2.80, 2.80, 2.80, 2.80]
```

### F5: moe_tp_experts output hook - ALL_REDUCE
```
GPU0: [0.57, 0.57, 0.57, 0.57]  ─┐
                                 ├──► all_reduce(SUM) ──► [3.37, 3.37, 3.37, 3.37]
GPU1: [2.80, 2.80, 2.80, 2.80]  ─┘

expert_out = [3.37, 3.37, 3.37, 3.37]  (same on both GPUs)
```

### F6: Apply routing weight
```
moe_out = 0.6 * [3.37, 3.37, 3.37, 3.37] = [2.02, 2.02, 2.02, 2.02]
```

---

## BACKWARD PASS

Assume `grad_output = [1.0, 1.0, 1.0, 1.0]`

### B1: Gradient through routing weight
```
grad_expert_out = 0.6 * [1, 1, 1, 1] = [0.6, 0.6, 0.6, 0.6]
```

### B2: moe_tp_experts output hook backward - IDENTITY
```
all_reduce_forward backward = identity
grad = [0.6, 0.6, 0.6, 0.6] passes through unchanged to both GPUs
```

### B3: down_proj backward (rowwise - no comm)
```
GPU0: grad_inter = grad @ down_0        GPU1: grad_inter = grad @ down_1
      = [.6,.6,.6,.6] @ [[.1,.1],             = [.6,.6,.6,.6] @ [[.1,.1],
                         [.1,.1],                                [.1,.1],
                         [.1,.1],                                [.1,.1],
                         [.1,.1]]                                [.1,.1]]
      = [0.24, 0.24]                          = [0.24, 0.24]
```

### B4: silu*up backward
```
GPU0: grad_gate ≈ [0.23, 0.23]           GPU1: grad_gate ≈ [0.48, 0.48]
      grad_up   ≈ [0.69, 0.69]                 grad_up   ≈ [1.68, 1.68]
      grad_gate_up = [0.23, 0.23,              grad_gate_up = [0.48, 0.48,
                      0.69, 0.69]                              1.68, 1.68]
```

### B5: gate_up_proj backward
```
GPU0: grad_x_0 = grad_gate_up @ gate_up_0     GPU1: grad_x_1 = grad_gate_up @ gate_up_1
      = [.23,.23,.69,.69] @ [[.1,.2,.3,.4],         = [.48,.48,1.68,1.68] @ [[.5,.6,.7,.8],
                             [.1,.2,.3,.4],                                  [.5,.6,.7,.8],
                             [.1,.1,.1,.1],                                  [.2,.2,.2,.2],
                             [.1,.1,.1,.1]]                                  [.2,.2,.2,.2]]
      ≈ [0.18, 0.28, 0.37, 0.46]                   ≈ [1.15, 1.25, 1.34, 1.44]
```

### B6: moe_tp_experts input hook backward - ALL_REDUCE
```
GPU0: [0.18, 0.28, 0.37, 0.46]  ─┐
                                 ├──► all_reduce(SUM) ──► [1.33, 1.53, 1.71, 1.90]
GPU1: [1.15, 1.25, 1.34, 1.44]  ─┘

grad_x = [1.33, 1.53, 1.71, 1.90]  (same on both GPUs)
```

---

## Summary

```
FORWARD:
x ──► gate_up (local) ──► silu*up (local) ──► down (local) ──► ALL_REDUCE ──► out
      [each GPU has                            [partial       [sum partials]
       half of                                  results]
       intermediate]

BACKWARD:
grad_out ──► identity ──► down bwd (local) ──► silu*up bwd ──► gate_up bwd ──► ALL_REDUCE ──► grad_x
                          [no comm for                         [local grad]   [sum grads]
                           rowwise bwd]
```

**Communication:**
- Forward: 1 all_reduce after expert computation
- Backward: 1 all_reduce for input gradient (+ 1 for routing weights gradient)
